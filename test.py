import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    mean_psnr = 0
    mean_snr = 0
    mean_psnr_fbp = 0
    mean_snr_fbp = 0
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('%04d: process image... %s' % (i, img_path))
        visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)
        mean_psnr = mean_psnr + visualizer.calc_PSNR(visuals)
        mean_snr = mean_snr + visualizer.calc_SNR(visuals)
        mean_psnr_fbp = mean_psnr_fbp + visualizer.calc_PSNR_fbp(visuals)
        mean_snr_fbp = mean_snr_fbp + visualizer.calc_SNR_fbp(visuals)
    mean_psnr = mean_psnr/len(data_loader)
    mean_snr = mean_snr/len(data_loader)
    mean_psnr_fbp = mean_psnr_fbp/len(data_loader)
    mean_snr_fbp = mean_snr_fbp/len(data_loader)
    print('The mean PSNR, SNR on {} data is {:.2f}, {:.2f}'.format(opt.phase,mean_psnr, mean_snr))
#    print('The mean SNR is {}\n'.format(mean_snr))
    print('The mean PSNR, SNR on {} data with FBP is {:.2f}, {:.2f}\n'.format(opt.phase, mean_psnr_fbp, mean_snr_fbp))
#    print('The mean SNR with FBP is {}\n'.format(mean_snr_fbp))
    webpage.save()
