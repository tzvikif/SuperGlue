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
import copy

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
                          avg_dist,
                          calcScores,
                          draw_match)

torch.set_grad_enabled(False)
# create a folder with same many samples.
# each sample consist of 3 images. every sample has different START_KPTS 
def init_params(vs):
    frame, ret,(orig_image_w,orig_image_h) = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'
    #superpoint image0
    frame_tensor = frame2tensor(frame, device)
    data = matching.superpoint({'image': frame_tensor})
    data0 = {k+'0': data[k] for k in keys}
    data0['image0'] = frame_tensor
    image_s = frame
    image_id = 0
    #superpoint image1
    frame, ret,(_,_) = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'
    stem0, stem1 = image_id, vs.i - 1
    frame_tensor = frame2tensor(frame, device)
    data = matching.superpoint({'image': frame_tensor})
    data1 = {k+'1': data[k] for k in keys}
    data1['image1'] = frame_tensor
    image_a = frame
    last_image_id = 1
    #superpoint image2
    frame, ret,(_,_) = vs.next_frame()
    frame_tensor = frame2tensor(frame, device)
    data = matching.superpoint({'image': frame_tensor})
    data2 = {k+'0': data[k] for k in keys}
    data2['image0'] = frame_tensor
    image_d = frame
    last_image_id = 2

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)
    if opt.text_output_dir is not None:
        print('==> Will write text outputs to {}'.format(opt.text_output_dir))
        #Path(opt.text_output_dir).mkdir(exist_ok=True)
    timer = AverageTimer()
    timer.update('data')
    #01 - source -> aux
    pred_sa = matching({**data0, **data1})
    kpts_s = data0['keypoints0'][0].cpu().numpy()
    kpts_a = data1['keypoints1'][0].cpu().numpy()
    indices_s = pred_sa['indices0'][0].cpu().numpy()
    #confidence01_to_0 = pred01['matching_scores0'][0].cpu().numpy()
    full_scores_sa = pred_sa['full_scores']
    full_scores_sa_wo_sinkhorn = pred_sa['full_scores_wo_sinkhon']

    full_scores_as = torch.transpose(full_scores_sa,2,1)
    full_scores_as_wo_sinkhorn = torch.transpose(full_scores_sa_wo_sinkhorn,2,1)

    #12 - aux -> destination
    pred_da = matching({**data1, **data2}) #data1 pref 1, data2 pref 0
    #kpts12_1 = data1['keypoints1'][0].cpu().numpy()
    kpts_d = data2['keypoints0'][0].cpu().numpy()
    #indices01_1 = pred21['indices1'][0].cpu().numpy()
    #confidence12_to_2 = pred21['matching_scores0'][0].cpu().numpy()
    full_scores_da = pred_da['full_scores']
    full_scores_da_wo_sinkhorn = pred_da['full_scores_wo_sinkhon']
    #20 - destination -> source
    data2 = None
    data = matching.superpoint({'image': frame_tensor})
    data2 = {k+'1': data[k] for k in keys}
    data2['image1'] = frame_tensor
    pred_sd = matching({**data0, **data2})
    #kpts20_0 = data0['keypoints0'][0].cpu().numpy()
    #kpts20_2 = data2['keypoints1'][0].cpu().numpy()
    #matches20_to_0 = pred02['indices0'][0].cpu().numpy()
    #confidence20_to_0 = pred02['matching_scores0'][0].cpu().numpy()
    full_scores_sd = pred_sd['full_scores']
    full_scores_sd_wo_sinkhorn = pred_sd['full_scores_wo_sinkhon']

    timer.update('forward')

    matching_as = {'kpts_s':kpts_s,'kpts_a':kpts_a,
    'full_scores':full_scores_as,
    'full_scores_wo_sinkhorn':full_scores_as_wo_sinkhorn}
    matching_sd = { 'kpts_s':kpts_s,'kpts_d':kpts_d,
    'full_scores':full_scores_sd,
    'full_scores_wo_sinkhorn':full_scores_sd_wo_sinkhorn}
    matching_da = {'kpts_d':kpts_d,'kpts_a':kpts_a,
    'full_scores':full_scores_da,
    'full_scores_wo_sinkhorn':full_scores_da_wo_sinkhorn}
    return {'matching_sd':matching_sd,
     'matching_da':matching_da,
     'matching_as':matching_as,
     'image_s':image_s,
     'image_d':image_d,'image_a':image_a,
     'orig_image_w':orig_image_w,'orig_image_h':orig_image_h,
     'indices_s':indices_s}

def scale_H(H,orig_image_size,new_image_size):
    orig_image_w = orig_image_size[0]
    orig_image_h = orig_image_size[1]
    new_image_w = new_image_size[0]
    new_image_h = new_image_size[1]
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
    return H

def draw_images_with_triangles(params,H,warped_kpts,tris,output_path,number_of_images=10,with_images=False):
    image_s = params['image_s']
    image_d = params['image_d']
    image_a = params['image_a']
    matching_sd = params['matching_sd']
    matching_da = params['matching_da']
    matching_as = params['matching_as']
    kpts_s = params['matching_sd']['kpts_s']
    indices_s = params['indices_s']

    #kpts_perm = np.random.permutation(len(kpts_sd_0))
    
    for idx,kpt_s_idx in enumerate( range(0,len(kpts_s)) ):
        text = list()
        #kpt_idx = 528
        #if idx == 60:
        #    break
        tri_out,text_matches = draw_triangles(tris,warped_kpts,kpt_s_idx,image_s,image_d,image_a)
        text.extend(text_matches)
        if opt.output_dir is not None:
            Path(output_path).mkdir(exist_ok=True)
            text_out_file_path = str(Path(output_path, 'kpts.txt'))
            write_to_file(text,text_out_file_path)
            if with_images==True:
                images_path = os.path.join(output_path,'triangles')
                Path(images_path).mkdir(exist_ok=True)
                stem = f'matches_{kpt_s_idx}'
                out_file = stem + '.png'
                cv2.imwrite(os.path.join(output_path,out_file), tri_out)
                cv2.destroyAllWindows()
def draw_improved_images(params_orig_list,tris_orig_list,params_imp1_list,params_imp2_list,tris_imp1_list,tris_imp2_list,output_paths):
    for params_orig,params_imp1,params_imp2,tris_orig,tris_imp1,tris_imp2,output_path in zip(params_orig_list,params_imp1_list,params_imp2_list,tris_orig_list,tris_imp1_list,tris_imp2_list,output_paths):
        Path(output_path).mkdir(exist_ok=True)
        image_s = params_orig['image_s']
        image_d = params_orig['image_d']
        matching_sd_orig = params_orig['matching_sd']['full_scores'][0,:,:]
        matching_sd_imp1 = params_imp1['matching_sd']['full_scores'][0,:,:]
        matching_sd_imp2 = params_imp2['matching_sd']['full_scores'][0,:,:]
        kpts_s = params_orig['matching_sd']['kpts_s']
        kpts_d= params_orig['matching_sd']['kpts_d']
        indices_sd_0 = params_orig['indices_s']
        warped_kpts = params_orig['warped_kpts']
        for kpt_image_s_idx in range(len(kpts_s)):
            max_idx_orig = np.argmax(matching_sd_orig[kpt_image_s_idx,:]).numpy()
            max_idx_imp1 = np.argmax(matching_sd_imp1[kpt_image_s_idx,:]).numpy()
            max_idx_imp2 = np.argmax(matching_sd_imp2[kpt_image_s_idx,:]).numpy()

            orig_match = {'kpts_s':kpts_s[kpt_image_s_idx],'kpts_d':kpts_d[max_idx_orig]}
            imp1_match = {'kpts_s':kpts_s[kpt_image_s_idx],'kpts_d':kpts_d[max_idx_imp1]}
            imp2_match = {'kpts_s':kpts_s[kpt_image_s_idx],'kpts_d':kpts_d[max_idx_imp2]}
            warped_kpt = warped_kpts[kpt_image_s_idx]
            kpts_d_orig = kpts_d[max_idx_orig]
            kpts_d_imp1 = kpts_d[max_idx_imp1]
            kpts_d_imp2 = kpts_d[max_idx_imp2]
            dist_orig_to_ground_truth = np.sqrt(np.dot(kpts_d_orig-warped_kpt,kpts_d_orig-warped_kpt))
            dist_imp1_to_ground_truth = np.sqrt(np.dot(kpts_d_imp1-warped_kpt,kpts_d_imp1-warped_kpt))
            dist_imp2_to_ground_truth = np.sqrt(np.dot(kpts_d_imp2-warped_kpt,kpts_d_imp2-warped_kpt))
            if dist_orig_to_ground_truth > dist_imp1_to_ground_truth:
                match_out = draw_match(image_s,image_d,orig_match,imp1_match,warped_kpt)
                stem = f'matches1_{kpt_image_s_idx}'
                out_file = str(Path(output_path, stem + '_comparison.png'))
                cv2.imwrite(out_file, match_out)
                '''
            if max_idx_orig != max_idx_imp2:
                out2 = draw_match(image_s,image_d,orig_match,imp2_match,warped_kpt)
                stem = f'matches2_{kpt_image_s_idx}'
                out_file = str(Path(output_path, stem + '.png'))
                out_file_test = str(Path(opt.output_dir, stem + '_test.png'))
                cv2.imwrite(out_file, match_out)
                '''
def evalError(new_params):
    is_imp = False
    if not new_params == None:
        is_imp = True
    root_dir = opt.input
    sub_dirs = os.listdir(root_dir)
    os.chdir(root_dir)
    total_dist,total_cnt = 0.0,0.0
    tris_list = list()
    params_list = list()
    warped_kpts_list = list()
    output_paths = list()
    sub_dirs = [sd for sd in sub_dirs if sd[0]!='.']
    for i,sub_dir in enumerate(sub_dirs):
        vs = VideoStreamer(sub_dir+'/', opt.resize, opt.skip,
                    opt.image_glob, max_length=3)
        file_name = [file for file in os.listdir(sub_dir) if file[0]=='H']
        file_name = file_name[0]
        if is_imp == False:
            params = init_params(vs)
        else:
            params = new_params[i]
        H = load_H(os.path.join(sub_dir,file_name))
        H = scale_H(H,(params['orig_image_w'],params['orig_image_h']),opt.resize)
        output_path = Path(os.path.join(opt.output_dir,sub_dir))
        if is_imp == True:
            output_path = Path(os.path.join(opt.output_dir,sub_dir+'_imp'))
        else:
            output_path = Path(os.path.join(opt.output_dir,sub_dir))
        output_paths.append(output_path)
        kpts_s = params['matching_sd']['kpts_s']
        warped_kpts = warp(kpts_s,H)
        
        tris = create_triangles(image_s=params['image_s'],image_d=params['image_d'],image_a=params['image_a'],
        matching_sd=params['matching_sd'],matching_da=params['matching_da'],matching_as=params['matching_as'])
        tris_list.append(tris)
        params['warped_kpts'] = warped_kpts
        params_list.append(params)
        draw_images_with_triangles(params,H,warped_kpts,tris,output_path,with_images=True)
        valid_indices = params['indices_s']
        dist,cnt = avg_dist(triangles=tris,warped_kpts=warped_kpts,valid_indices=valid_indices)
        total_dist+= dist
        total_cnt+= cnt
        warped_kpts_list.append(warped_kpts)
    vs.cleanup()
    os.chdir('..')
    return params_list,tris_list,output_paths,total_dist/total_cnt
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
    orig_params_list,tris_list,output_paths,base_error = evalError(None)
    valid_indices_list = [x['indices_s'] for x in orig_params_list]
    full_scores_list = [x['matching_sd']['full_scores'] for x in orig_params_list]
    output = f'base error:{base_error}:' 
    print(output)
    new_scores_list,new_scores_sh_list = calcScores(tris_list,full_scores_list,valid_indices_list)
    #update params with new scores
    params1 = copy.deepcopy(orig_params_list)
    params2 = copy.deepcopy(orig_params_list)
    sg = matching.superglue 
    for i,(new_scores,new_scores_sh) in enumerate(zip(new_scores_list,new_scores_sh_list)):
        params1[i]['matching_sd']['full_scores'] = new_scores
        a = sg.sh(new_scores_sh) 
        params2[i]['matching_sd']['full_scores'] = a
    _,tris1,_,error1 = evalError(params1)
    _,tris2,_,error2 = evalError(params2)
    print(f'error1:{error1}')
    print(f'error2:{error2}')
    draw_improved_images(orig_params_list,tris_list,params1,params2,tris1,tris2,output_paths)

    






    