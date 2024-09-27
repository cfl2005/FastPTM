import tensorflow as tf
import numpy as np
import threading
import time

def MyLoop(coord,worker_id):
    # Use the collaborative tool provided by the tf. coordinator class to determine whether the current thread has terminated
    while nor coord.should_stop():
        # Randomly stop all threads
        if np.random.rand() < 0.1:
            # coord.request_stop()
            coord.request_stop()
        else:
            print("working on id:%d \n" % worker_id)
        time.sleep(1)\

# Declars a tf.train.coordinator class to collaborate on a thread
coord = tf.train.Coordinator()

# Declare the creation of 5 threads
threads = [
 threading.Thread(target=MyLoop,args(coord,i)) for i in range(5)
]

# Start all threads
for t in threads: t.start()

coord.join(threads)