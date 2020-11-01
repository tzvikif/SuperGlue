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
import os
from pathlib import Path

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor,
                          make_matching_plot_one_to_many,
                          create_triangles,
                          draw_triangles,
                          write_to_file,
                          write_warped_kpts,
                          warp,
                          load_H,
                          avg_dist)

torch.set_grad_enabled(False)
# create a folder with same many samples.
# each sample consist of 3 images. every sample has different START_KPTS 
def ex1(vs,H,output_path):
    frame, ret,(orig_image_w,orig_image_h) = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'
    #superpoint image0
    frame_tensor = frame2tensor(frame, device)
    data = matching.superpoint({'image': frame_tensor})
    data0 = {k+'0': data[k] for k in keys}
    data0['image0'] = frame_tensor
    image0 = frame
    image_id = 0
    #superpoint image1
    frame, ret,(_,_) = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'
    stem0, stem1 = image_id, vs.i - 1
    frame_tensor = frame2tensor(frame, device)
    data = matching.superpoint({'image': frame_tensor})
    data1 = {k+'1': data[k] for k in keys}
    data1['image1'] = frame_tensor
    image1 = frame
    last_image_id = 1
    #superpoint image2
    frame, ret,(_,_) = vs.next_frame()
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
        #Path(opt.text_output_dir).mkdir(exist_ok=True)
    timer = AverageTimer()
    timer.update('data')
    #01
    pred01 = matching({**data0, **data1})
    kpts01_0 = data0['keypoints0'][0].cpu().numpy()
    kpts01_1 = data1['keypoints1'][0].cpu().numpy()
    indices01_0 = pred01['indices0'][0].cpu().numpy()
    #confidence01_to_0 = pred01['matching_scores0'][0].cpu().numpy()
    full_scores01 = pred01['full_scores']
    full_scores01_wo_sinkhorn = pred01['full_scores_wo_sinkhon']
    #12
    pred21 = matching({**data1, **data2}) #data1 pref 1, data2 pref 0
    kpts12_1 = data1['keypoints1'][0].cpu().numpy()
    kpts12_2 = data2['keypoints0'][0].cpu().numpy()
    #indices01_1 = pred21['indices1'][0].cpu().numpy()
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
    #matches20_to_0 = pred02['indices0'][0].cpu().numpy()
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
    tris = create_triangles(image0,image2,image1,matching02,matching21,matching10)
    new_image_w,new_image_h = opt.resize
    x_scale = new_image_w/orig_image_w 
    y_scale = new_image_h/orig_image_h
    l_scale = np.array([[1/x_scale,0,0],
                    [0,1/y_scale,0],
                    [0,0,1]],dtype=float)
    r_scale = np.array([[x_scale,0,0],
                    [0,y_scale,0],
                    [0,0,1]],dtype=float)
    H = np.matmul(H,l_scale)
    H = np.matmul(r_scale,H)
    warped_kpts = warp(kpts01_0,H)
    dist,cnt = avg_dist(triangles=tris,warped_kpts=warped_kpts,valid_indices=indices01_0)
    for idx,kpt_idx in enumerate(kpts_perm):
        text = list()
        if idx == 3:
            break
        tri_out,text_matches = draw_triangles(tris,warped_kpts,kpt_idx,image0,image2,image1)
        text.extend(text_matches)
        if opt.output_dir is not None:
            Path(output_path).mkdir(exist_ok=True)
            text_out_file_path = str(Path(output_path, 'kpts.txt'))
            write_to_file(text,text_out_file_path)
            #stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
            stem = f'matches_{kpt_idx}'
            out_file = str(Path(output_path, stem + '.png'))
            out_file_test = str(Path(opt.output_dir, stem + '_test.png'))
            print('\nWriting image to {}'.format(out_file))
            cv2.imwrite(out_file, tri_out)
    
    cv2.destroyAllWindows()
    vs.cleanup()
    return dist,cnt

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

    root_dir = opt.input
    sub_dirs = os.listdir(root_dir)
    os.chdir(root_dir)
    total_dist,total_cnt = 0.0,0.0
    for sub_dir in sub_dirs:
        if sub_dir[0] == '.':
            continue
        vs = VideoStreamer(sub_dir+'/', opt.resize, opt.skip,
                       opt.image_glob, max_length=3)
        file_name = [file for file in os.listdir(sub_dir) if file[0]=='H']
        file_name = file_name[0]
        #os.chdir(sub_dir)
        H = load_H(os.path.join(sub_dir,file_name))
        output_path = Path(os.path.join(opt.output_dir,sub_dir))
        dist,cnt = ex1(vs,H,str(output_path))
        total_dist+= dist
        total_cnt+= total_cnt
    output = f'avg err:{total_dist/total_cnt} avg err after correction:' 
    