# coding=utf-8
import paddle.fluid as fluid
from PIL import Image
import numpy as np
from PIL import Image, ImageTk # 导入图像处理函数库
import tkinter as tk           # 导入GUI界面函数库
import os
import tkinter
import tkinter.filedialog
import time
from PIL import ImageGrab
import matplotlib as mpl
import matplotlib.pyplot as plt


#截图部分
class FreeCapture():
    """ 用来显示全屏幕截图并响应二次截图的窗口类
    """
    def __init__(self, root, img):
        
        #变量X和Y用来记录鼠标左键按下的位置
        self.X = tkinter.IntVar(value=0)
        self.Y = tkinter.IntVar(value=0)
        #屏幕尺寸
        screenWidth = root.winfo_screenwidth()
        screenHeight = root.winfo_screenheight()
        #创建顶级组件容器
        self.top = tkinter.Toplevel(root, width=screenWidth, height=screenHeight)
        #不显示最大化、最小化按钮
        self.top.overrideredirect(True)
        self.canvas = tkinter.Canvas(self.top,bg='white', width=screenWidth, height=screenHeight)
        #显示全屏截图，在全屏截图上进行区域截图 
        self.image = tkinter.PhotoImage(file=img)
        self.canvas.create_image(screenWidth//2, screenHeight//2, image=self.image)
        
        self.lastDraw = None
        #鼠标左键按下的位置
        def onLeftButtonDown(event):
            self.X.set(event.x)
            self.Y.set(event.y)
            #开始截图
            self.sel = True

        self.canvas.bind('<Button-1>', onLeftButtonDown)

        def onLeftButtonMove(event):
            
            #鼠标左键移动，显示选取的区域
            if not self.sel:
                return
            try: #删除刚画完的图形，要不然鼠标移动的时候是黑乎乎的一片矩形
                self.canvas.delete(self.lastDraw)
            except Exception as e:
                pass
            self.lastDraw = self.canvas.create_rectangle(self.X.get(), self.Y.get(), event.x, event.y, outline='green')
            
        def onLeftButtonUp(event):
            
            #获取鼠标左键抬起的位置，保存区域截图
            self.sel = False
            try:
                self.canvas.delete(self.lastDraw)
            except Exception as e:
                pass

            time.sleep(0.5)
            #考虑鼠标左键从右下方按下而从左上方抬起的截图
            left, right = sorted([self.X.get(), event.x])
            top, bottom = sorted([self.Y.get(), event.y])
            pic = ImageGrab.grab((left+1, top+1, right, bottom))
         
            timestamp=0
            timestamp = time.time()
            timeArray = time.localtime(timestamp)
            global formatTime
            formatTime = time.strftime("%Y_%m_%d_%H_%M_%S", timeArray)   
            
            pic_path = 'E:\\1\\' + formatTime + '.png'
            
            
            if pic_path:
                pic.save(pic_path)
                
            #关闭当前窗口
            self.top.destroy()
            

        self.canvas.bind('<B1-Motion>', onLeftButtonMove) # 按下左键
        self.canvas.bind('<ButtonRelease-1>', onLeftButtonUp) # 抬起左键
        #让canvas充满窗口，并随窗口自动适应大小
        self.canvas.pack(fill=tkinter.BOTH, expand=tkinter.YES)
        


def screenShot():
    
    """ 自由截屏的函数 (button按钮的事件)
    """
#    print("test")
    root.state('icon')  # 最小化主窗体
    time.sleep(0.2)
    im = ImageGrab.grab()
    # 暂存全屏截图
    im.save('temp.png')
    im.close()
    # 进行自由截屏 
    w = FreeCapture(root, 'temp.png')
    button_screenShot.wait_window(w.top)
    
    # 截图结束，恢复主窗口，并删除temp.png文件
    root.state('normal')
    os.remove('temp.png')
   
   

####
root = tkinter.Tk()
root.title('自由截屏')
#指定窗口的大小
root.geometry('200x200')
#不允许改变窗口大小
root.resizable(False,False)

# ================== 布置截屏按钮 ====================================
button_screenShot = tkinter.Button(root, text='截取细胞', command=screenShot)
button_screenShot.place(relx=0.25, rely=0.25, relwidth=0.5, relheight=0.5)
# ================== 完 =============================================

counter=0
try:
    root.mainloop()
    counter+=1

except:
    root.destroy()+1

# 创建执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
# 保存预测模型路径
save_path = '40infer_model/'

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
    
# 获取图片数据
#此处需要修改地址
img = load_image('E:\\1\\' + formatTime + '.png')

# 执行预测
result = exe.run(program=infer_program,
                 feed={feeded_var_names[0]: img},
                 fetch_list=target_var)
# 显示图片并输出结果最大的label
lab = np.argsort(result)[0][0][-1]

names = ['EOSINOPHIL', 'LYMPHOCYTE','MONOCYTE','NEUTROPHIL']

print('lab :%d, name :%s, accurancy: %f' % (lab, names[lab], result[0][0][lab]))


#GUI部分
# 创建窗口 设定大小并命名
window = tk.Tk()
window.title('CITP')
window.geometry('1000x1000')
global img_png           # 定义全局变量 图像的
var = tk.StringVar()    # 这时文字变量储存器

# 创建打开图像和显示图像函数
def Open_Img():
    global img_png
    var.set('Already opened')
    Img = Image.open('E:\\1\\' + formatTime + '.png')
    img_png = ImageTk.PhotoImage(Img)

def Open_CITP():
    global img_png
    var.set('Already opened')
    Img = Image.open('E:/1/CITP.png')
    img_png = ImageTk.PhotoImage(Img)
    label_Img = tk.Label(window, image=img_png)
    label_Img.pack()
def Show_Img():
    global img_png
    var.set('Already shown')   
    label_Img = tk.Label(window, image=img_png)
    label_Img.pack()

def Show_Result():
    var.set('The result is \n name :%s, accurancy: %f' % (names[lab], result[0][0][lab]))

Open_CITP()
# 创建文本窗口，显示当前操作状态
Label_Show = tk.Label(window,
    textvariable=var,   # 使用 textvariable 替换 text, 因为这个可以变化
    bg='white', font=('Arial', 12), width=50, height=2)
Label_Show.pack()
# 创建打开图像按钮
btn_Open = tk.Button(window,
    text='open the image',      # 显示在按钮上的文字
    width=15, height=2,
    command=Open_Img)     # 点击按钮式执行的命令
btn_Open.pack()    # 按钮位置
# 创建显示图像按钮
btn_Show = tk.Button(window,
    text='show the image',      # 显示在按钮上的文字
    width=15, height=2,
    command=Show_Img)     # 点击按钮式执行的命令
btn_Show.pack()    # 按钮位置



# 创建显示结果按钮
btn_Result = tk.Button(window,
    text='show the result',      # 显示在按钮上的文字
    width=15, height=2,
    command=Show_Result)     # 点击按钮式执行的命令
btn_Result.pack()    # 按钮位置
# 运行整体窗口
window.mainloop()

