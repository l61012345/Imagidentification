# coding=utf-8
import paddle.fluid as fluid
from PIL import Image
import numpy as np
import os.path
# 创建执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
# 保存预测模型路径
save_path = '160infer_model/'

# 从模型中获取预测程序、输入数据名称列表、分类器
[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)
# 预处理图片
def load_image(file):
    img = Image.open(file)
    # 统一图像大小
    img = img.resize((224, 224), Image.ANTIALIAS)
    # 转换成numpy值
    img = np.array(img).astype(np.float32)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# 执行预测
rootdir= 'D://paddle//TEST_SIMPLE//EOSINOPHIL'
for parent,dirnames,filenames in os.walk(rootdir):
    for filename in filenames:
     imgpath=os.path.join(parent,filename)
     img = load_image(imgpath)
     result = exe.run(program=infer_program,
                 feed={feeded_var_names[0]: img},
                 fetch_list=target_var)
     # 显示图片并输出结果最大的label
     lab = np.argsort(result)[0][0][-1]
     names = ['EOSINOPHIL', 'LYMPHOCYTE','MONOCYTE','NEUTROPHIL']
     print('filename: %s ,lab :%d, name :%s, accurancy: %f' % (filename,lab, names[lab], result[0][0][lab]))
