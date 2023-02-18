from optparse import OptionParser
from task import Task
from utils import logging_utils
from model_param_space import param_space_dict
import datetime
import config
import os


def parse_args(parser):
    parser.add_option('-m', '--model', dest='model_name', type='string')
    parser.add_option('-d', '--data', dest='data_name', type='string')
    parser.add_option('-p', '--portion', dest='portion', type=int, default=100)
    parser.add_option('-a', '--alpha', dest='alpha', type=float, default=0.)
    parser.add_option('-o', '--savename', dest='save_name', type='string', default='')
    parser.add_option('-r', '--runs', dest='runs', type='int', default=2)
    parser.add_option('-g', '--gpu', dest='gpu', type='string', default="0")
    options, args = parser.parse_args()
    return options, args


def main(options):
    time_str = datetime.datetime.now().isoformat()
    if len(options.save_name) == 0:
        log_name = 'Eval_[Model@%s]_[Data@%s]_%s.log' % (options.model_name,
                                                         options.data_name, time_str)
    else:
        log_name = 'Eval_[Model@%s]_[Data@%s]_%s.log' % (options.save_name,
                                                         options.data_name, time_str)
    logger = logging_utils.get_logger(config.LOG_DIR, log_name)
    params_dict = param_space_dict[options.model_name]
    # params_dict['num_epochs'] = max(params_dict['e_1'] + params_dict['e_2'] + params_dict['e_3'],
    #                                 params_dict['num_epochs'])
    for ancestor_rate in [0.4,0.45,0.5,0.55,0.6,0.65]:
        for label_smoothing in [0.1,0.15,0.2,0.25,0.3]:
            for loss_rate in [0.8]:
                for ssl_rate in [1.0]:
                    for confidence in [0.75]:
                        for only_single in [True,False]:
                            if ancestor_rate>=1-label_smoothing:
                                continue
                            if ssl_rate == 0.0 and confidence != 0.9:
                                continue
                            # params_dict['state_size'] =300
                            # params_dict['wpe_dim'] = 20
                            params_dict['num_epochs'] = 40
                            params_dict['save_rate'] = 0.5
                            params_dict['label_smoothing']=label_smoothing
                            params_dict['ancestor_rate'] = ancestor_rate
                            params_dict['loss_rate'] = loss_rate
                            params_dict['alpha'] = 0.0
                            params_dict['ssl_loss_rate'] = ssl_rate
                            params_dict['confidence'] = confidence
                            params_dict['only_single'] = only_single

                            task = Task(model_name=options.model_name, data_name=options.data_name, cv_runs=options.runs,
                                        params_dict=params_dict, logger=logger, portion=options.portion,
                                        save_name=options.save_name)

                            print('-' * 50 + 'refit' + '-' * 50)
                            task.refit()


if __name__ == '__main__':
    t_parser = OptionParser()
    opt, _ = parse_args(t_parser)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    main(opt)
