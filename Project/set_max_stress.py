# Python script to recompute the stress/color levels in images from this dataset,
# using '-m' (= max stress level) parameter:
#
# - pixels in images with _MAX_ over limit will take the most dark red color (the highest stress), based the color bar attached here,
# - stress values in all the images then can be compared via colors of their pixels (the same color = the same stress), except the pixels over limit,
# - colors (of the pixels in the images) are the same as the color bar's colors attached here,
# - the script supports a resume function. Just stop & run it again and select the same I/O directories.
#
# Command line arguments:
# -----------------------
#
# -m <value>   recompute fea stress image for maximal stress of <value> MPa
# -h           this help
#
# Example:
# --------
# python set_max_stress.py -m 300
#
# 
#   ***************************
#   *        LICENSE          *
#   * (c) 2020 GNU LGPL v3    *
#   ***************************
#
#   **********************************
#   *            AUTHOR              *
#   * Jaroslav Matej, MSc, Ph.D.     *
#   * Technical University in Zvolen *
#   * Faculty of Technology          *
#   * jaroslav.matej@tuzvo.sk        *
#   * matej.tuzvo@gmail.com          *
#   **********************************
#
#   DOI: 


import cv2
import os
import numpy as np
import sys, getopt
import easygui
import shutil

###################################
#                                 #
#      FUNCTIONS & CLASSES        #
#                                 #
###################################

def get_number_of_files(dir):
    if os.path.exists(dir):
        count_f = len(os.listdir(dir))
        print("All the files: " + str(count_f))
        return count_f
    else:
        return None

def progressbar(it, prefix="", sizepg=60, filepg=sys.stdout):
    countpg = len(it)
    def show(j):
        x = int(sizepg*j/countpg)
        filepg.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(sizepg-x), j, countpg))
        filepg.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    filepg.write("\n")
    filepg.flush()

def load_color_bar():
    # load color bar
    color_bar_img = cv2.imread('color_bar.png')
    rows, cols, _ = color_bar_img.shape
    print(str(rows) + ", " + str(cols))
    # count colors
    colors_count = 0
    colors = []
    prev_color = 0
    for col in color_bar_img[0]:
        if np.all(col == prev_color):
            continue
        else:
            colors_count += 1
            prev_color = col
            colors.append(col)
    return colors

def recompute_image(file_image_in, inp_dir, cbar_colors, max_stress):
    # read min, max from file name e.g. '00ade60ceeb211e98c7b0862664f9d1c_MIN_0.0_MAX_164.8_OUT.png'
    min_idx_start = file_image_in.find("_MIN_")
    max_idx_start = file_image_in.find("_MAX_")
    out_idx = file_image_in.find("_OUT.png")
    if min_idx_start == -1 or max_idx_start == -1 or out_idx == -1:
        return None
    min_str = file_image_in[min_idx_start + 5 : max_idx_start]
    max_str = file_image_in[max_idx_start + 5 : out_idx]
    min_value = float(min_str)
    max_value = float(max_str)

    # modify pixels' colors and create new image
    fea_stress_img = cv2.imread(os.path.join(inp_dir, file_image_in))   # load from full path
    rows, cols, _ = fea_stress_img.shape
    fea_output_img = 255* np.ones((rows, cols, 3), np.uint8)

    for pty in range(rows):
        for ptx in range(cols):
            pnt = fea_stress_img[pty, ptx]
            idx_in_cbar = -1
            for idx in range(len(cbar_colors)):
                if pnt[0] == cbar_colors[idx][0] and pnt[1] == cbar_colors[idx][1] and pnt[2] == cbar_colors[idx][2]:
                    idx_in_cbar = idx
                    # recompute pixel's value
                    new_idx = idx_in_cbar / max_stress * max_value
                    if new_idx >= len(cbar_colors):
                        new_idx = len(cbar_colors) - 1
                    # write to the new image
                    if color_mode == 'color':
                        fea_output_img[pty, ptx] = cbar_colors[int(new_idx)]
                    break

    # save the picture
    save_recomputed_pic = os.path.join(output_dir, file_image_in)
    cv2.imwrite(save_recomputed_pic, fea_output_img)

    #cv2.imshow("FEA Recomputed Image", fea_output_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #####################################################
 #######################################################
#########################################################
#                                                       #
#                                                       #
#                     MAIN APP                          #
#                                                       #
#                                                       #
#########################################################
 #######################################################
  #####################################################

###################################
#                                 #
#           VARIABLES             #
#                                 #
###################################

max_stress_level = -1  # MPa
color_mode = 'color'


###################################
#                                 #
#           MAIN LOOP             #
#                                 #
###################################

# load desired maximal stress level as command line argument with options:

if len(sys.argv) == 1:
    print("Script requires arguments. See -h option.")
    exit(1)

try:
    opts, args = getopt.getopt(sys.argv[1:],"hm:c:")
except getopt.GetoptError:
    print('set_max_level.py -m <max stress level in MPa>')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print("\n# Python script to recompute the color levels in images from this dataset,\n# using '-m' (= max stress level) parameter:\n#\n# - pixels in images with _MAX_ over limit will take the most dark red color, based the color bar attached here\n# - stress values in all the images than can be compared via colors of its pixels (the same color = the same stress), except the pixels over limit\n# - colors are the same as the color bar's colors attached here\n#\n# Command line arguments:\n# -----------------------\n#\n# -m <value>   recompute fea stress image for maximal stress of <value> MPa\n# -h           this help\n#\n# Example:\n# --------\n# python set_max_stress.py -m 300\n#\n# The script supports a resume function. Just stop & run it again and select the same I/O directories.\n#\n")
        sys.exit()
    elif opt in "-m":
        try:
            max_stress_level = float(str(arg))
        except ValueError:
            print("Input number only for -m option")
            exit(1)

if max_stress_level < 0 and color_mode == 'color':
    print("Script requires arguments. See -h option.")
    exit(1)

if color_mode == 'color':
    print("FEA Images will be recomputed to " + str(max_stress_level) + " MPa")

# select input & output dirs using easygui
while True:
    input_dir = easygui.diropenbox('SELECT --INPUT-- DIRECTORY')
    print(input_dir)
    files_count = get_number_of_files(input_dir)
    if files_count == 0:
        continue
    if files_count is not None:
        break
while True:
    output_dir = easygui.diropenbox('SELECT --OUTPUT-- DIRECTORY')
    print(output_dir)
    if os.path.exists(output_dir):
        break

# load images
all_files = os.listdir(input_dir)   # returns only file names

# load color bar colors
color_bar_colors = load_color_bar()

#
#  process files in pairs only
#

for i1 in progressbar(all_files, "Converting: "):

    if "_INP.png" in i1:
        continue

    i2 = i1.replace("_OUT.png", "_INP.png")

    i1_fp = os.path.join(input_dir, i1)     # fullpaths: _OUT.png
    i2_fp = os.path.join(input_dir, i2)     # _INP.png

    if not os.path.exists(i1_fp) or not os.path.exists(i2_fp):
        continue

    o1_fp = os.path.join(output_dir, i1)    # output files - fullpaths
    o2_fp = os.path.join(output_dir, i2)

    if os.path.exists(o1_fp) and os.path.exists(o2_fp):
        continue

    # copy _INP file
    shutil.copy(i2_fp, output_dir)        

    # recompute image's color pixels
    recompute_image(i1, input_dir, color_bar_colors, max_stress_level)
    







