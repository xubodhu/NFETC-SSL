from model import Model
import tensorflow as tf
from utils import data_utils, prior_utils
from utils import eval_utils
import numpy as np
import config
from functools import reduce
import copy

tf.set_random_seed(seed=config.RANDOM_SEED)

class NFETC_LS(Model):
    def __init__(self, num_train, sequence_length, mention_length, num_classes, vocab_size,
                 embedding_size, position_size, pretrained_embedding, wpe, type_info, hparams, dataset):
        self.data_name = dataset
        self.sequence_length = sequence_length
        self.mention_length = mention_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.position_size = position_size
        self.pretrained_embedding = pretrained_embedding
        self.wpe = wpe
        self.num_train = num_train

        self.state_size = hparams.state_size
        self.hidden_layers = hparams.hidden_layers
        self.hidden_size = hparams.hidden_size
        self.wpe_dim = hparams.wpe_dim
        self.l2_reg_lambda = hparams.l2_reg_lambda
        self.lr = hparams.lr

        self.dense_keep_prob = hparams.dense_keep_prob
        self.rnn_keep_prob = hparams.rnn_keep_prob

        self.hp = hparams
        self.batch_size = hparams.batch_size
        self.num_epochs = hparams.num_epochs
        #
        self.label_smoothing = hparams.label_smoothing
        self.filter = hparams.filter
        self.ancestor_rate = hparams.ancestor_rate
        self.ast_tune = prior_utils.create_prior(type_info,1.0)

        self.loss_rate = hparams.loss_rate
        self.confidence = hparams.confidence
        self.ssl_loss_rate = hparams.ssl_loss_rate

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.build()


    def add_placeholders(self):
        self.input_words = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_words')
        self.input_textlen = tf.placeholder(tf.int32, [None], name='input_textlen')
        self.input_mentions = tf.placeholder(tf.int32, [None, self.mention_length], name='input_mentions')
        self.input_mentionlen = tf.placeholder(tf.int32, [None], name='input_mentionlen')
        self.input_positions = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_positions')
        self.input_labels = tf.placeholder(tf.float32, [None, self.num_classes], name='input_labels')
        self.input_ids = tf.placeholder(tf.int32, [None], name='input_ids')
        self.phase = tf.placeholder(tf.bool, name='phase')
        self.dense_dropout = tf.placeholder(tf.float32, name='dense_dropout')
        self.rnn_dropout = tf.placeholder(tf.float32, name='rnn_dropout')

        #
        self.label_root = tf.placeholder(tf.float32,[None, self.num_classes], name='input_label_root')
        #
        #
        self.ols_labels = tf.placeholder(tf.float32,[None, self.num_classes], name='input_ols_labels')
        #
        #
        self.mask = tf.placeholder(tf.float32, [None], name = 'single_data_mask')
        #

        tmp = [i for i in range(self.mention_length)]
        tmp[0] = self.mention_length
        interval = tf.Variable(tmp, trainable=False)
        interval_row = tf.expand_dims(interval, 0)
        upper = tf.expand_dims(self.input_mentionlen - 1, 1)
        mask = tf.less(interval_row, upper)
        self.mention = tf.where(mask, self.input_mentions, tf.zeros_like(self.input_mentions))
        self.mentionlen = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)
        self.mentionlen = tf.cast(
            tf.where(tf.not_equal(self.mentionlen, tf.zeros_like(self.mentionlen)), self.mentionlen,
                     tf.ones_like(self.mentionlen)), tf.float32)
        self.mentionlen = tf.expand_dims(self.mentionlen, 1)


    def create_feed_dict(self, input_words, input_textlen, input_mentions, input_mentionlen, input_positions,
                         input_labels=None, input_ids=None, phase=False, dense_dropout=1., rnn_dropout=1.,input_label_root=None,ols_labels_batch = None, mask_batch = None):
        feed_dict = {
            self.input_words: input_words,
            self.input_textlen: input_textlen,
            self.input_mentions: input_mentions,
            self.input_mentionlen: input_mentionlen,
            self.input_positions: input_positions,
            self.phase: phase,
            self.dense_dropout: dense_dropout,
            self.rnn_dropout: rnn_dropout
        }
        if input_labels is not None:
            feed_dict[self.input_labels] = input_labels
        if input_ids is not None:
            feed_dict[self.input_ids] = input_ids
        #
        if input_label_root is not None:
            feed_dict[self.label_root] = input_label_root
        #
        #
        if ols_labels_batch is not None:
            feed_dict[self.ols_labels] = ols_labels_batch
        #
        #
        if mask_batch is not None:
            feed_dict[self.mask] = mask_batch
        #

        return feed_dict

    #
    def add_embedding(self):
        with tf.device('/cpu:0'), tf.name_scope('word_embedding'):
            W = tf.Variable(self.pretrained_embedding, trainable=False, dtype=tf.float32, name='W')
            self.embedded_words = tf.nn.embedding_lookup(W, self.input_words)
            self.embedded_mentions = tf.nn.embedding_lookup(W, self.input_mentions)
            self.mention_embedding = tf.divide(tf.reduce_sum(tf.nn.embedding_lookup(W, self.mention),
                                                             axis=1), self.mentionlen)

        with tf.device('/cpu:0'), tf.name_scope('position_embedding'):
            W = tf.Variable(self.wpe, trainable=False, dtype=tf.float32, name='W')
            self.wpe_chars = tf.nn.embedding_lookup(W, self.input_positions)
        self.input_sentences = tf.concat([self.embedded_words, self.wpe_chars], 2)

    #
    def add_hidden_layer(self, x, idx):
        dim = self.feature_dim if idx == 0 else self.hidden_size
        with tf.variable_scope('hidden_%d' % idx):
            W = tf.get_variable('W', shape=[dim, self.hidden_size],
                                initializer=tf.contrib.layers.xavier_initializer(seed=config.RANDOM_SEED))
            b = tf.get_variable('b', shape=[self.hidden_size],
                                initializer=tf.contrib.layers.xavier_initializer(seed=config.RANDOM_SEED))
            h = tf.nn.xw_plus_b(x, W, b)
            h_norm = tf.layers.batch_normalization(h, training=self.phase)
            h_drop = tf.nn.dropout(tf.nn.relu(h_norm), self.dense_dropout, seed=config.RANDOM_SEED)
        return h_drop

    #
    def extract_last_relevant(self, outputs, seq_len):
        batch_size = tf.shape(outputs)[0]
        max_length = int(outputs.get_shape()[1])
        num_units = int(outputs.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (seq_len - 1)
        flat = tf.reshape(outputs, [-1, num_units])
        relevant = tf.gather(flat, index)
        return relevant

    #
    def add_prediction_op(self):
        self.add_embedding()
        self.bsize = tf.shape(self.embedded_mentions)[0]

        with tf.name_scope('sentence_repr'):
            attention_w = tf.get_variable('attention_w', [self.state_size, 1])
            cell_forward = tf.contrib.rnn.LSTMCell(self.state_size)
            cell_backward = tf.contrib.rnn.LSTMCell(self.state_size)
            cell_forward = tf.contrib.rnn.DropoutWrapper(cell_forward, input_keep_prob=self.dense_dropout,
                                                         output_keep_prob=self.rnn_dropout, seed=config.RANDOM_SEED)
            cell_backward = tf.contrib.rnn.DropoutWrapper(cell_backward, input_keep_prob=self.dense_dropout,
                                                          output_keep_prob=self.rnn_dropout, seed=config.RANDOM_SEED)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_forward, cell_backward, self.input_sentences,
                sequence_length=self.input_textlen, dtype=tf.float32)
            outputs_added = tf.nn.tanh(tf.add(outputs[0], outputs[1]))
            alpha = tf.nn.softmax(tf.reshape(tf.matmul(
                tf.reshape(outputs_added, [-1, self.state_size]),
                attention_w),
                [-1, self.sequence_length]))
            alpha = tf.expand_dims(alpha, 1)
            self.sen_repr = tf.squeeze(tf.matmul(alpha, outputs_added))

        with tf.name_scope('mention_repr'):
            cell = tf.contrib.rnn.LSTMCell(self.state_size)
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dense_dropout,
                                                 output_keep_prob=self.rnn_dropout, seed=config.RANDOM_SEED)

            outputs, states = tf.nn.dynamic_rnn(
                cell, self.embedded_mentions,
                sequence_length=self.input_mentionlen, dtype=tf.float32)
            self.men_repr = self.extract_last_relevant(outputs, self.input_mentionlen)

        self.features = tf.concat([self.sen_repr, self.men_repr, self.mention_embedding], -1)
        self.feature_dim = self.state_size * 2 + self.embedding_size

        # visual representation
        self.h_output = self.features

        h_drop = tf.nn.dropout(tf.nn.relu(self.features), self.dense_dropout, seed=config.RANDOM_SEED)
        h_drop.set_shape([None, self.feature_dim])
        h_output = tf.layers.batch_normalization(h_drop, training=self.phase)

        # get representation layer
        for i in range(self.hidden_layers):
            h_output = self.add_hidden_layer(h_output, i)
        if self.hidden_layers == 0:
            self.hidden_size = self.feature_dim

        with tf.variable_scope('typeVec', reuse=tf.AUTO_REUSE):
            W = tf.get_variable('W', shape=[self.hidden_size, self.num_classes],
                                initializer=tf.contrib.layers.xavier_initializer(
                                    seed=config.RANDOM_SEED))  # hidden size= 660
            b = tf.get_variable('b', shape=[self.num_classes],
                                initializer=tf.contrib.layers.xavier_initializer(seed=config.RANDOM_SEED))


            self.scores = tf.nn.xw_plus_b(h_output, W, b, name='scores')  # [batch,num class]
            self.proba = tf.nn.softmax(self.scores, name='proba')

            # hier
            #self.adjusted_proba = tf.matmul(self.proba, self.tune)
            self.adjusted_proba = tf.clip_by_value(self.proba, 1e-10, 1.0, name='adprob')

            #
            self.maxtype = tf.argmax(self.proba, 1, name='maxtype')
            #
            self.predictions = tf.one_hot(self.maxtype, self.num_classes, name='prediction')

    #
    def add_loss_op(self):

        with tf.name_scope("loss"):
            prob_s, prob_w = tf.split(self.adjusted_proba, 2)
            #
            target_index = self.input_labels + self.label_root + self.label_smoothing / self.num_classes
            #
            self.batch_losses = -tf.reduce_sum(target_index * tf.log(prob_s), 1)
            losses = tf.reduce_mean(self.batch_losses * (1-self.mask))
            #losses = tf.reduce_mean(-tf.reduce_sum(target_index * tf.log(self.adjusted_proba), 1))
            self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda), weights_list=tf.trainable_variables())

            self.ols_losses = -tf.reduce_sum(self.ols_labels * tf.log(prob_s), 1)

            #
            pseudo_labels = tf.stop_gradient(prob_w)
            ssl_losses = -tf.reduce_sum(pseudo_labels * tf.log(prob_s), 1)
            pseudo_mask = tf.to_float(tf.reduce_max(pseudo_labels, axis=1) >= self.confidence)
            self.ssl_loss =  tf.reduce_mean(ssl_losses * pseudo_mask)

            #all loss
            self.loss = self.loss_rate * tf.reduce_mean(losses) + (1-self.loss_rate)*tf.reduce_mean(self.ols_losses * (1-self.mask)) + self.l2_loss + \
                        self.ssl_loss_rate*self.ssl_loss

    #
    def add_training_op(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.grads_and_vars = optimizer.compute_gradients(self.loss)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

    #
    def train_on_batch(self, sess, input_words, input_textlen, input_mentions,
                       input_mentionlen, input_positions, input_labels, input_ids,input_label_root,ols_labels_batch, mask_batch):
        feed = self.create_feed_dict(input_words, input_textlen, input_mentions, input_mentionlen, input_positions,
                                     input_labels, input_ids, True, self.dense_keep_prob, self.rnn_keep_prob,input_label_root,ols_labels_batch,mask_batch)
        Variablelist = [self.train_op, self.global_step, self.loss, self.l2_loss, self.batch_losses,self.adjusted_proba, self.ssl_loss]

        a = sess.run(Variablelist, feed_dict=feed)
        step = a[1]
        if step and step % 100 == 0:
            print('step {}, loss {:g} l2_loss {:g} SSL_loss {:g}'.format(step, a[2], a[3],a[6]))
        return a[4],a[5]

    #
    def get_scores(self, preds, labels, id2type):
        label_path = eval_utils.label_path
        if type(preds) == np.ndarray:
            preds = [[label_path(id2type[i]) for i, x in enumerate(line) if x > 0] for line in preds]
            preds = [list(set(reduce(lambda x, y: x + y, line))) for line in preds]
        else:
            preds = [label_path(id2type[x]) for x in preds]

        def vec2type(v):
            s = []
            for i in range(len(v)):
                if v[i]:
                    s.extend(label_path(id2type[i]))
            return set(s)

        labels_test = [vec2type(x) for x in labels]  # path will caculate the father node for strict acc
        acc = eval_utils.strict(labels_test, preds)
        _, _, macro = eval_utils.loose_macro(labels_test, preds)
        _, _, micro = eval_utils.loose_micro(labels_test, preds)

        return acc, macro, micro

    #
    def predict(self, sess, test):
        batches = data_utils.batch_iter(test, self.batch_size, 1, shuffle=False)
        all_predictions = []
        all_labels = []
        all_maxtype = []
        h_output = []
        for batch in batches:
            words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, labels_batch = zip(*batch)

            feed = self.create_feed_dict(words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch)
            # add visual
            batch_predictions, batchmaxtype, batch_h_output = sess.run([self.predictions, self.maxtype, self.h_output], feed_dict=feed)
            if len(all_predictions) == 0:
                all_predictions = batch_predictions
            else:
                all_predictions = np.concatenate([all_predictions, batch_predictions])
            if len(all_maxtype) == 0:
                all_maxtype = batchmaxtype
            else:
                all_maxtype = np.concatenate([all_maxtype, batchmaxtype])

            if len(all_labels) == 0:
                all_labels = np.array(labels_batch)
            else:
                all_labels = np.concatenate([all_labels, np.array(labels_batch)])

            if len(h_output) ==0:
                h_output = np.array(batch_h_output)
            else:
                h_output = np.concatenate([h_output, batch_h_output])


        return all_predictions, all_maxtype, h_output

    #
    def get_label_root(self,input_labels):
        batch_size = len(input_labels)
        new_input_labels = np.zeros((batch_size, self.num_classes))
        label_root = np.zeros((batch_size, self.num_classes))
        for i in range(batch_size):
            ast = np.zeros(self.num_classes)
            for j in range(self.num_classes):
                if input_labels[i][j] > 0.9:
                    ast += self.ast_tune[j,:]
                else:
                    continue
            ast = np.where(ast > 0.9, 1.0, 0.0)
            w = self.ancestor_rate / np.sum(ast)
            label_root[i, :] = w * ast
            #
            new_input_labels[i,:] = ((1-self.label_smoothing-self.ancestor_rate)/np.sum(input_labels[i]))*input_labels[i]

        return label_root, new_input_labels
    #

    def get_ols_label(self,input_labels,ols_matrix):
        batch_size = len(input_labels)
        ols_labels = np.zeros((batch_size, self.num_classes))
        for i in range(batch_size):
            ols_label = np.zeros(self.num_classes)
            num = 0
            for j in range(self.num_classes):
                if input_labels[i][j]>0.99:
                    ols_label = ols_label + ols_matrix[j,:]
                    num+=1
            ols_label = ols_label / num
            if np.abs(np.sum(ols_label) - 1.0) > 1e-5 and np.sum(ols_label)> 1e-5:
                print('probs sum:',np.sum(ols_label))
                raise ValueError('OLS label sum error')
            #
            ols_labels[i,:] = ols_label
        return ols_labels

    def evaluate(self, sess, train_batches,epoch_num=-1,init_ols_matrix = None):
        epoch_losses = None
        epoch_ids = None
        #
        if init_ols_matrix is None:
            ols_matrix = np.zeros((self.num_classes, self.num_classes)) + (1e-10)
        else:
            ols_matrix = init_ols_matrix
        #
        for batch in train_batches:

            words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, labels_batch, ids_batch, mask_batch = zip(
                *batch)
            #
            label_root_batch,new_batch_labels = self.get_label_root(labels_batch)
            #
            #
            ols_labels_batch = self.get_ols_label(labels_batch,ols_matrix)
            #
            #
            words_batch = words_batch + copy.deepcopy(words_batch)
            textlen_batch = textlen_batch + copy.deepcopy(textlen_batch)
            mentions_batch = mentions_batch + copy.deepcopy(mentions_batch)
            mentionlen_batch = mentionlen_batch + copy.deepcopy(mentionlen_batch)
            positions_batch = positions_batch + copy.deepcopy(positions_batch)
            #

            batch_losses,batch_prob = self.train_on_batch(sess, words_batch, textlen_batch, mentions_batch, mentionlen_batch,
                                positions_batch, new_batch_labels, ids_batch, label_root_batch, ols_labels_batch, mask_batch)
            batch_losses = batch_losses[:len(ids_batch)]
            batch_prob = batch_prob[:len(ids_batch)]
            if epoch_ids is None:
                epoch_ids = ids_batch
                epoch_losses = np.squeeze(batch_losses)
            else:
                epoch_ids = np.concatenate((epoch_ids,ids_batch))
                epoch_losses = np.concatenate((epoch_losses,np.squeeze(batch_losses)))
            #
            batch_pred = np.argmax(np.array(batch_prob),axis=1)
            for i in range(batch_pred.shape[0]):
                pred = batch_pred[i]
                #
                if labels_batch[i][pred] > 0.99 and batch_prob[i][pred] >= self.hp.tao:#and np.sum(labels_batch[i]) == 1.0:
                    ols_matrix[pred,:] = ols_matrix[pred,:] + batch_prob[i]

            ols_matrix = ols_matrix / np.sum(ols_matrix,axis=1,keepdims=True)

        print('epoch losses shape:',epoch_losses.shape)
        print('epoch ids shape:',epoch_ids.shape)
        print(f"online ls matrix:\n{ols_matrix}")

        #np.save(f'./exp/{self.data_name}_{epoch_num}',ols_matrix)

        return epoch_ids,epoch_losses


    def evaluate_batch(self,sess,batch):
        words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, labels_batch, ids_batch = zip(
            *batch)
        #
        label_root_batch, new_batch_labels = self.get_label_root(labels_batch)
        #
        batch_losses = self.train_on_batch(sess, words_batch, textlen_batch, mentions_batch, mentionlen_batch,
                                           positions_batch, new_batch_labels, ids_batch, label_root_batch)
        batch_ids = np.array(ids_batch)
        batch_losses = np.squeeze(batch_losses)
        return batch_ids,batch_losses

