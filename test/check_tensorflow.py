
import tensorflow as tf
import numpy as np

def main():
    arr = [13,13,12,15]
    style_ids_x = arr
    style_ids_y = arr
    style_ids_alpha = [0,0,0,0]
    
    _testOfreduce_sum(style_ids_x)

def _testOfreduce_sum(style_ids_x):
    print(style_ids_x)
    sum_op = tf.reduce_sum(style_ids_x)
    less_op = tf.less(sum_op,0)

    style_embedding_np = np.random.uniform(-1,1,(256,100))
    embedding_op = tf.nn.embedding_lookup(style_embedding_np,style_ids_x)

    random_op = np.random_uniform((4, 100), -1, 1).astype(np.float32)

    with tf.Session() as sess:
        print(sess.run([sum_op]))
        print(sess.run([less_op]))
        true_res = sess.run([random_op])
        #true_res = np.array(true_res)
        #print(true_res.shape)


        false_res = sess.run([embedding_op])
        false_res = np.array(false_res)
        print(false_res.shape)

        char_z = tf.one_hot(style_ids_x, 56)

        z = tf.concat([true_res, char_z], axis=1)
        print(z.shape)




if __name__ == "__main__":
    main()

