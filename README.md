# captchaCnn
python homework about captcha recognize

CAPTCHA注册码识别实践

本实验中实现了基于TensorFlow的注册码识别任务。
如后所附代码，本实验中有captcha_gen.py用于生成captcha图片并返回图片中的真实验证码信息，cnn_train.py用于训练参数，captcha_cnn.py用于测试训练的模型，util.py包含了一些所需要的函数。

一、重点说明一下cnn_train.py内容：
这里进行了三层卷积神经网络计算，分别是卷积层、池化层、全链接层。
另外还有优化函数和准确度计算函数。
def weight_variable(shape, w_alpha=0.01):
    """
    增加噪音，随机生成权重
    :param shape:
    :param w_alpha:
    :return:
    """
    initial = w_alpha * tf.random_normal(shape)
    return tf.Variable(initial)


def bias_variable(shape, b_alpha=0.1):
    """
    增加噪音，随机生成偏置项
    :param shape:
    :param b_alpha:
    :return:
    """
    initial = b_alpha * tf.random_normal(shape)
    return tf.Variable(initial)


def conv2d(x, w):
    """
    局部变量线性组合，步长为1，模式‘SAME’代表卷积后图片尺寸不变，即零边距
    :param x:
    :param w:
    :return:
    """
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """
    max pooling,取出区域内最大值为代表特征， 2x2pool，图片尺寸变为1/2
    :param x:
    :return:
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



训练函数：
def cnn_graph(x, keep_prob, size, captcha_list=CAPTCHA_LIST, captcha_len=CAPTCHA_LEN):
    """
    三层卷积神经网络计算图
    :param x:
    :param keep_prob:
    :param size:
    :param captcha_list:
    :param captcha_len:
    :return:
    """
    # 图片reshape为4维向量
    image_height, image_width = size
    # 使用-1进行调整
    x_image = tf.reshape(x, shape=[-1, image_height, image_width, 1])

    # layer 1
    # filter定义为3x3x1， 输出32个特征, 即32个filter
    w_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    # rulu激活函数
    h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x_image, w_conv1), b_conv1))
    # 池化
    h_pool1 = max_pool_2x2(h_conv1)
    # dropout防止过拟合
    h_drop1 = tf.nn.dropout(h_pool1, keep_prob)

    # layer 2
    w_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_drop1, w_conv2), b_conv2))
    h_pool2 = max_pool_2x2(h_conv2)
    h_drop2 = tf.nn.dropout(h_pool2, keep_prob)

    # layer 3
    w_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(tf.nn.bias_add(conv2d(h_drop2, w_conv3), b_conv3))
    h_pool3 = max_pool_2x2(h_conv3)
    h_drop3 = tf.nn.dropout(h_pool3, keep_prob)

    # full connect layer
    image_height = int(h_drop3.shape[1])
    image_width = int(h_drop3.shape[2])
    w_fc = weight_variable([image_height*image_width*64, 1024])
    b_fc = bias_variable([1024])
    h_drop3_re = tf.reshape(h_drop3, [-1, image_height*image_width*64])
    h_fc = tf.nn.relu(tf.add(tf.matmul(h_drop3_re, w_fc), b_fc))
    h_drop_fc = tf.nn.dropout(h_fc, keep_prob)

    # out layer
    w_out = weight_variable([1024, len(captcha_list)*captcha_len])
    b_out = bias_variable([len(captcha_list)*captcha_len])
    y_conv = tf.add(tf.matmul(h_drop_fc, w_out), b_out)
    return y_conv



正式训练中，将每次训练时准确度超过acc_rate（初始时0.9）的记录下来，没记录一次acc_rate自增0.01，当达到0.95时退出训练，即总共有非连续的5次测试正确率超过了acc_rate时退出并保存最好的模型。

def train(height=CAPTCHA_HEIGHT, width=CAPTCHA_WIDTH, y_size=len(CAPTCHA_LIST)*CAPTCHA_LEN):
    '''
    cnn训练
    :param height:
    :param width:
    :param y_size:
    :return:
    '''
    # cnn在图像大小是2的倍数时性能最高, 如果图像大小不是2的倍数，可以在图像边缘补无用像素
    # 在图像上补2行，下补3行，左补2行，右补2行
    # np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))

    acc_rate = 0.9
    # 按照图片大小申请占位符
    x = tf.placeholder(tf.float32, [None, height * width])
    y = tf.placeholder(tf.float32, [None, y_size])
    # 防止过拟合 训练时启用 测试时不启用
    keep_prob = tf.placeholder(tf.float32)
    # cnn模型
    y_conv = cnn_graph(x, keep_prob, (height, width))
    # 最优化
    optimizer = optimize_graph(y, y_conv)
    # 偏差
    accuracy = accuracy_graph(y, y_conv)
    # 启动会话.开始训练
    saver = tf.train.Saver()
    sess = tf.Session()
    #初始化
    sess.run(tf.global_variables_initializer())
    step = 0
    while 1:
        batch_x, batch_y = next_batch(64)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.75})
        # 每训练一百次测试一次
        if step % 100 == 0:
            batch_x_test, batch_y_test = next_batch(100)
            acc = sess.run(accuracy, feed_dict={x: batch_x_test, y: batch_y_test, keep_prob: 1.0})
            print(datetime.now().strftime('%c'), ' step:', step, ' accuracy:', acc)
            # 偏差满足要求，保存模型
            if acc > acc_rate:
                model_path = os.getcwd() + os.sep + str(acc_rate) + "captcha.model"
                saver.save(sess, model_path, global_step=step)
                acc_rate += 0.01
                #模型要求设置为0.85节省时间，应设为0.99
                if acc_rate > 0.95:
                    break
        step += 1
    sess.close()


二、开始预测函数captcha_cnn
这里需要注意的是我们产生的验证码是彩色的，我们不需要，所以压迫转灰度图。
def convert2gray(img):
    """
    图片转为黑白，3维转1维
    :param img:
    :return:
    """
    if len(img.shape) > 2:
        img = np.mean(img, -1)
    return img

以上是一种简单的方法，十分好用。

前期训练得到的数据存储为文件，在预测时调用。

预测结果不一定准确，本实验中最好的准确度设为0.94，并不算好，所以在预测时有时还是有错误。
