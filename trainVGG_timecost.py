import os
import shutil
import vgg
import paddle as paddle
import reader
import paddle.fluid as fluid
import datetime

#每次测试需要输入的信息
#图片的类别数目,细胞分类是4类
types=4
#训练的次数
train=200
#训练模型保存的地址，接着当前目录下的地址写就行
save_path='infer_models/vgg/100_round/'
#trainer.list的位置
trainer_path='E:/paddle/list/trainer.list'
#test.list的位置
test_path='E:/paddle/list/test.list'

#设置裁切照片的参数
crop_size = 224
resize_size = 250

# 定义输入层
image = fluid.layers.data(name='image', shape=[3, crop_size, crop_size], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
# 获取分类器，因为这次只爬取了4个类别的图片，所以分类器的类别大小为4
model = vgg.vgg_bn_drop(image, types)
# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

# 获取训练和测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-3,
                                          regularization=fluid.regularizer.L2DecayRegularizer(1e-4))
opts = optimizer.minimize(avg_cost)
# 获取自定义数据
train_reader = paddle.batch(reader=reader.train_reader(trainer_path, crop_size, resize_size), batch_size=32)
test_reader = paddle.batch(reader=reader.test_reader(test_path, crop_size), batch_size=32)

# 定义一个使用GPU的执行器
place = fluid.CUDAPlace(0)
# CPU 执行器 place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

for pass_id in range(train):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):
        # 第一个时间戳
        time1=(datetime.datetime.now())
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, acc])

        # 每100个batch打印一次信息
        if batch_id % 100 == 0:
            # 第二个时间戳
            time2=(datetime.datetime.now())
            timecost=time2-time1
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f, Timecost:' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]) ,end=' ')
            print(timecost)

    # 进行测试
    test_accs = []
    test_costs = []
    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=test_program,
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_cost, acc])
        test_accs.append(test_acc[0])
        test_costs.append(test_cost[0])
    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))
    test_acc = (sum(test_accs) / len(test_accs))
    print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))
    # 删除旧的模型文件
    shutil.rmtree(save_path, ignore_errors=True)
    # 创建保持模型文件目录
    os.makedirs(save_path)
    # 保存预测模型
    fluid.io.save_inference_model(save_path, feeded_var_names=[image.name], target_vars=[model], executor=exe)
