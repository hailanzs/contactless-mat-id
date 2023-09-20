from multiprocessing import Process, Queue
import os
import logging
import time
import glob
import natsort
from prod_dca import save_data_1843_jan12
import vlc
import scipy.io as sio

range_res = 0.056266129623825

def consumer(q, index):
    pass

def main(exp_num,exp_object, date,num_frms,sound_name,sleep_time,sleep_time_sound, sound_path, exp_path):

    num_producers = 2
    num_consumers = 1
    max_queue_size = 10000

    logging.info(f"Execution happening with {num_producers} producers and {num_consumers} consumers")

    q_main = Queue(maxsize=max_queue_size)

    producers = []
    consumers = []
    # Create our producer processes by passing the producer function and it's arguments
    producers.append(Process(target=save_data_1843_jan12, args=(q_main, 0, exp_num,exp_object,date,num_frms,sleep_time, exp_path)))
    producers.append(Process(target=play_sound, args=(q_main, 1, sound_name, exp_num, date, sleep_time_sound, sound_path, exp_path)))

    
    # Create consumer processes
    for i in range(0, num_consumers):
        p = Process(target=consumer, args=(q_main, 0))
        p.daemon = False

        consumers.append(p)

    # Start the producers and consumer
    # The Python VM will launch new independent processes for each Process object
    for p in producers:
        p.start()
 
    for c in consumers:
        c.start()
 
    # Like threading, we have a join() method that synchronizes our program
    try:
        for p in producers:
            p.join()

        # Wait for empty chunks queue to exit
        for c in consumers:
            c.join()
    except KeyboardInterrupt as ki:
        print(f"Program terminated by keyboard")

    logging.info('Parent process exiting...')
    # update status to done, or cancelled or error?


def play_sound(sound_name, exp_num, date, sleep_time_sound, sound_path, exp_path):

    if(sound_name != "brown noise" and sound_name != "white pink brown noise" and sound_name!= "nothing"):
        time.sleep(0.1)
        sound_start = time.time()
        print("starting chirp: " + str(sound_start))
        print(sound_name)    
	# plays the sound specified by sound_path and sound name
	# requires vlc package and VLC installed (installation instructions online)
        player = vlc.MediaPlayer(os.path.join(sound_path, sound_name) + ".mp3")
        player.play()
        time.sleep(sleep_time_sound)
        sound_end = time.time()
        print("ending chirp: " + str(sound_end))
        exp_path = os.path.join(exp_path, date)

        if not os.path.isdir(exp_path):
            os.mkdir(exp_path)
	# saves the sound start and end times in the same mat file as the data
        my_dict = sio.loadmat(os.path.join(exp_path, date, str(exp_num) + ".mat"))
        my_dict["sound_start"] = sound_start
        my_dict["sound_end"] = sound_end
        sio.savemat(os.path.join(exp_path, date, str(exp_num) + ".mat"), my_dict)


def find_exp_num(date, exp_path):
    
    if(len(natsort.natsorted(glob.glob(os.path.join(exp_path, date, '*.mat')))) <= 0):
        last_filename = 0
    else:
        last_file = natsort.natsorted(glob.glob(os.path.join(exp_path, date, '*.mat')))[-1]
        last_filename = os.path.basename(last_file)
        last_filename = int(os.path.splitext(last_filename)[0]) + 1
    return last_filename

if __name__ == "__main__":

    # CHANGE HERE
    iterations = 1
    exp_object = "brass"
    date = "oct-6"
    #######################################
    # paths to update
    dir_path = os.path.dirname(os.path.realpath('prod-dca.py'))
    sound_path = dir_path
    exp_path = os.path.join(dir_path, 'measured_data',)

    ######################################
    num_frms = 3750
    details = ""
    sleep_time = 0.5
    sleep_time_sound = 10.25
    exp_num = find_exp_num(date, exp_path)
    sounds = ["sine-test-calib-long"]
    num_exps = (iterations)*len(sounds) - 1
    ii = 0

    if not os.path.isdir(exp_path):
        os.mkdir(exp_path)

    if False: # if want to take picture of setup prior, set True
        camera = cv2.VideoCapture(0)
        for i in range(2):
            return_value, image = camera.read()
            cv2.imwrite(os.path.join(exp_path, date, str(exp_num) + "_" + str(exp_num+num_exps) + "_" + exp_object +  ".jpg"), image)
        del(camera)

    # for sound_name in sounds:
    for iter in range(iterations):
        sio.savemat(os.path.join(exp_path, date, str(exp_num+ii) + ".mat"), {"details": details})
        ii += 1

    for sound_name in sounds:
        for iter in range(iterations):
            main(exp_num, exp_object, date,num_frms, sound_name, sleep_time,sleep_time_sound, sound_path, exp_path)
            exp_num += 1
    print("next start number: ", str(exp_num))


