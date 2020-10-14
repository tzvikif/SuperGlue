#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import numpy as np

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor,
                          make_matching_plot_one_to_many,
                          create_triangles,
                          draw_triangles,
                          write_to_file)

torch.set_grad_enabled(False)
# create a folder with same many samples.
# each sample consist of 3 images. every sample has different START_KPTS 
def ex1(vs,opt):
    frame, ret = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'
    #superpoint image0
    frame_tensor = frame2tensor(frame, device)
    data = matching.superpoint({'image': frame_tensor})
    data0 = {k+'0': data[k] for k in keys}
    data0['image0'] = frame_tensor
    image0 = frame
    image_id = 0
    #superpoint image1
    frame, ret = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'
    stem0, stem1 = image_id, vs.i - 1
    frame_tensor = frame2tensor(frame, device)
    data = matching.superpoint({'image': frame_tensor})
    data1 = {k+'1': data[k] for k in keys}
    data1['image1'] = frame_tensor
    image1 = frame
    last_image_id = 1
    #superpoint image2
    frame, ret = vs.next_frame()
    frame_tensor = frame2tensor(frame, device)
    data = matching.superpoint({'image': frame_tensor})
    data2 = {k+'0': data[k] for k in keys}
    data2['image0'] = frame_tensor
    image2 = frame
    last_image_id = 2

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)
    if opt.text_output_dir is not None:
        print('==> Will write text outputs to {}'.format(opt.text_output_dir))
        Path(opt.text_output_dir).mkdir(exist_ok=True)
    timer = AverageTimer()
    timer.update('data')
    #01
    pred01 = matching({**data0, **data1})
    kpts01_0 = data0['keypoints0'][0].cpu().numpy()
    kpts01_1 = data1['keypoints1'][0].cpu().numpy()
    matches01_to_0 = pred01['matches0'][0].cpu().numpy()
    confidence01_to_0 = pred01['matching_scores0'][0].cpu().numpy()
    full_scores01 = pred01['full_scores']
    full_scores01_wo_sinkhorn = pred01['full_scores_wo_sinkhon']
    #12
    pred21 = matching({**data1, **data2}) #data1 pref 1, data2 pref 0
    kpts12_1 = data1['keypoints1'][0].cpu().numpy()
    kpts12_2 = data2['keypoints0'][0].cpu().numpy()
    matches12_to_2 = pred21['matches0'][0].cpu().numpy()
    confidence12_to_2 = pred21['matching_scores0'][0].cpu().numpy()
    full_scores12 = torch.transpose(pred21['full_scores'],2,1)
    full_scores12_wo_sinkhorn = torch.transpose(pred21['full_scores_wo_sinkhon'],2,1)
    #20
    data2 = None
    data = matching.superpoint({'image': frame_tensor})
    data2 = {k+'1': data[k] for k in keys}
    data2['image1'] = frame_tensor
    pred02 = matching({**data0, **data2})
    kpts20_0 = data0['keypoints0'][0].cpu().numpy()
    kpts20_2 = data2['keypoints1'][0].cpu().numpy()
    matches20_to_0 = pred02['matches0'][0].cpu().numpy()
    confidence20_to_0 = pred02['matching_scores0'][0].cpu().numpy()
    full_scores20 = torch.transpose(pred02['full_scores'],2,1) 
    full_scores20_wo_sinkhorn = torch.transpose(pred02['full_scores_wo_sinkhon'],2,1) 

    timer.update('forward')
    
    matching01 = {'kpts_s':kpts01_0,'kpts_d':kpts01_1,
    'full_scores':full_scores01,
    'full_scores_wo_sinkhorn':full_scores01_wo_sinkhorn}
    matching20 = {'kpts_s':kpts20_2,'kpts_d':kpts20_0,
    'full_scores':full_scores20,
    'full_scores_wo_sinkhorn':full_scores20_wo_sinkhorn}
    matching12 = {'kpts_s':kpts12_1,'kpts_d':kpts12_2,
    'full_scores':full_scores12,
    'full_scores_wo_sinkhorn':full_scores12_wo_sinkhorn}

    matching10 = {'kpts_s':kpts01_1,'kpts_d':kpts01_0,
    'full_scores':full_scores01.transpose(2,1),
    'full_scores_wo_sinkhorn':full_scores01_wo_sinkhorn.transpose(2,1)}
    matching02 = {  'kpts_s':kpts20_0,'kpts_d':kpts20_2,
    'full_scores':full_scores20.transpose(2,1),
    'full_scores_wo_sinkhorn':full_scores20_wo_sinkhorn.transpose(2,1)}
    matching21 = {'kpts_s':kpts12_2,'kpts_d':kpts12_1,
    'full_scores':full_scores12.transpose(2,1),
    'full_scores_wo_sinkhorn':full_scores12_wo_sinkhorn.transpose(2,1)}
    kpts_perm = np.random.permutation(len(kpts01_0))
    text = list()
    #text_file_path = 'text_output/kpts.txt'
    for idx,kpt_idx in enumerate(kpts_perm):
        #tri_out,text_matches = draw_triangles(image0,image2,image1,
        #matching02,matching21,matching10,for_kpt=kpt_idx)
        tri_out,text_matches = draw_triangles(image0,image1,image2,
        matching01,matching12,matching20,for_kpt=kpt_idx)
        text.extend(text_matches)
        if opt.text_output_dir is not None:
            text_out_file_path = str(Path(opt.text_output_dir, 'kpts.txt'))
            write_to_file(text,text_out_file_path)
        if opt.output_dir is not None:
            #stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
            stem = f'matches_{kpt_idx}'
            out_file = str(Path(opt.output_dir, stem + '.png'))
            print('\nWriting image to {}'.format(out_file))
            cv2.imwrite(out_file, tri_out)
    
    cv2.destroyAllWindows()
    vs.cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default='0',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')
    parser.add_argument(
        '--text_output_dir', type=str, default=None,
        help='Directory where to write output text file (If None, no output)')
    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)
    

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']

    vs = VideoStreamer(opt.input, opt.resize, opt.skip,
                       opt.image_glob, opt.max_length)
    # 1.
    # create a folder with same many samples.
    # each sample consist of 3 images. every sample has different START_KPTS 
    ex1(vs,opt)
    # 2. check accuracy with H matrix.
    '''
    frame, ret = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'

    frame_tensor = frame2tensor(frame, device)
    data = matching.superpoint({'image': frame_tensor})
    data0 = {k+'0': data[k] for k in keys}
    data0['image0'] = frame_tensor
    image0 = frame
    image_id = 0

    frame, ret = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'
    stem0, stem1 = image_id, vs.i - 1
    frame_tensor = frame2tensor(frame, device)
    data = matching.superpoint({'image': frame_tensor})
    data1 = {k+'1': data[k] for k in keys}
    data1['image1'] = frame_tensor
    image1 = frame
    last_image_id = 1

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    if not opt.no_display:
        cv2.namedWindow('SuperGlue matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SuperGlue matches', (640*2, 480))
    else:
        print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    print('==> Keyboard control:\n'
          '\tn: select the current frame as the anchor\n'
          '\te/r: increase/decrease the keypoint confidence threshold\n'
          '\td/f: increase/decrease the match filtering threshold\n'
          '\tk: toggle the visualization of keypoints\n'
          '\tq: quit')

    timer = AverageTimer()
    jump = True
    step = 0
    while True:
        if jump:
            for _ in range(step):
                frame, ret = vs.next_frame()
        frame, ret = vs.next_frame()
        jump = not jump
        stem2 = vs.i - 1
        if not ret:
            print('Finished demo_superglue.py')
            break
        assert ret, 'Error when reading the first frame (try different --input?)'
        #super point for image 2
        frame_tensor = frame2tensor(frame, device)
        data = matching.superpoint({'image': frame_tensor})
        data2 = {k+'0': data[k] for k in keys}
        data2['image0'] = frame_tensor
        image2 = frame

        timer.update('data')
        #01
        pred01 = matching({**data0, **data1})

        kpts01_0 = data0['keypoints0'][0].cpu().numpy()
        kpts01_1 = data1['keypoints1'][0].cpu().numpy()
        matches01_to_0 = pred01['matches0'][0].cpu().numpy()
        confidence01_to_0 = pred01['matching_scores0'][0].cpu().numpy()
        full_scores01 = pred01['full_scores']
        #12
        pred21 = matching({**data1, **data2}) #data1 pref 1, data2 pref 0
        kpts12_1 = data1['keypoints1'][0].cpu().numpy()
        kpts12_2 = data2['keypoints0'][0].cpu().numpy()
        matches12_to_2 = pred21['matches0'][0].cpu().numpy()
        confidence12_to_2 = pred21['matching_scores0'][0].cpu().numpy()
        full_scores12 = torch.transpose(pred21['full_scores'],2,1)
        #20
        data2 = None
        data = matching.superpoint({'image': frame_tensor})
        data2 = {k+'1': data[k] for k in keys}
        data2['image1'] = frame_tensor
        pred02 = matching({**data0, **data2})
        kpts20_0 = data0['keypoints0'][0].cpu().numpy()
        kpts20_2 = data2['keypoints1'][0].cpu().numpy()
        matches20_to_0 = pred02['matches0'][0].cpu().numpy()
        confidence20_to_0 = pred02['matching_scores0'][0].cpu().numpy()
        full_scores20 = torch.transpose(pred02['full_scores'],2,1) 
        #full_scores02 = pred02['full_scores'] 

        timer.update('forward')

        #valid = matches > -1
        #mkpts0 = kpts0[valid]
        #mkpts1 = kpts1[matches[valid]]
        
        #color = cm.jet(confidence01_0[valid])
        
        matching01 = {'kpts_s':kpts01_0,'kpts_d':kpts01_1,
        'full_scores':full_scores01}
        matching20 = {'kpts_s':kpts20_2,'kpts_d':kpts20_0,
        'full_scores':full_scores20}
        matching12 = {'kpts_s':kpts12_1,'kpts_d':kpts12_2,
        'full_scores':full_scores12}

        matching10 = {'kpts_s':kpts01_1,'kpts_d':kpts01_0,
        'full_scores':full_scores01.transpose(2,1)}
        matching02 = {'kpts_s':kpts20_0,'kpts_d':kpts20_2,
        'full_scores':full_scores20.transpose(2,1)}
        matching21 = {'kpts_s':kpts12_2,'kpts_d':kpts12_1,
        'full_scores':full_scores12.transpose(2,1)}
        
        if  jump:    #was a jump
            tri_out = draw_triangles(image0,image2,image1,
            matching02,matching21,matching10,for_kpt=np.random.randint(0,kpts20_0.shape[0]))
        else:
            tri_out = draw_triangles(image0,image1,image2,
            matching01,matching12,matching20,for_kpt=45)
        data1 = data2.copy()
        data0 = {k[:-1]+'0': data1[k] for k in data1.keys()}
        image0 = image1.copy()
        image1 = image2.copy()
        if not opt.no_display:
            cv2.imshow('SuperGlue matches', out)
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == 'q':
                vs.cleanup()
                print('Exiting (via q) demo_superglue.py')
                break
            elif key == 'n':  # set the current frame as anchor
                last_data = {k+'0': pred[k+'1'] for k in keys}
                last_data['image0'] = frame_tensor
                last_frame = frame
                last_image_id = (vs.i - 1)
            elif key in ['e', 'r']:
                # Increase/decrease keypoint threshold by 10% each keypress.
                d = 0.1 * (-1 if key == 'e' else 1)
                matching.superpoint.config['keypoint_threshold'] = min(max(
                    0.0001, matching.superpoint.config['keypoint_threshold']*(1+d)), 1)
                print('\nChanged the keypoint threshold to {:.4f}'.format(
                    matching.superpoint.config['keypoint_threshold']))
            elif key in ['d', 'f']:
                # Increase/decrease match threshold by 0.05 each keypress.
                d = 0.05 * (-1 if key == 'd' else 1)
                matching.superglue.config['match_threshold'] = min(max(
                    0.05, matching.superglue.config['match_threshold']+d), .95)
                print('\nChanged the match threshold to {:.2f}'.format(
                    matching.superglue.config['match_threshold']))
            elif key == 'k':
                opt.show_keypoints = not opt.show_keypoints

        timer.update('viz')
        timer.print()

        if opt.output_dir is not None:
            #stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
            stem = 'matches_{:02}_{:02}_{:02}'.format(stem0, stem1,stem2)
            out_file = str(Path(opt.output_dir, stem + '.png'))
            print('\nWriting image to {}'.format(out_file))
            cv2.imwrite(out_file, tri_out)

    cv2.destroyAllWindows()
    vs.cleanup()
    '''
