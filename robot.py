import pygame
import sys
import random
import math
import numpy as np
vec = pygame.math.Vector2
from math import *

# TODO
# 旋转方向与实际情况相反？
# 视野角度仅在0～180度间适用

pygame.init()

SIZE = WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
BACKGROUND_COLOR = (255,255,255)

screen = pygame.display.set_mode(SIZE)
robots = []
lights = []

# 机器人配置
VMAX = 15           # 最大速度
D = 1               # 轮子直径     
ANGLE = 0          # 水平眼睛抬起角度    # 30, -30, 0, 45,-45
VISION_ANGLE = 2*(90-ANGLE) # 视野角度  
mp4 = True         # 是否截图
RUNNING_TIME = 500 # 运行时间

# 颜色
WHITE = (255, 255, 255) 
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
DARKGRAY = (40, 40, 40)

# for bp algorithm
angles = []         # 两只眼睛与光源夹角
velocities = []     # 速度

# ANN 结构
input_size = 4
hidden_size = 20
output_size = 2
std = 1e-2








# 随机初始化方向
def get_rand_vec(dims):
    x = np.random.standard_normal(dims)
    r = np.sqrt((x*x).sum())
    return x / r

# 加载光源图片
class Light(object):
    def __init__(self,pos,img_src,id):
        self.image_light = pygame.image.load(img_src)
        # rect_light = image_light.get_rect()
        self.pos = np.array(pos)
        self.id = id
    def draw(self):
        screen.blit(self.image_light,self.pos)


# 绘制光源图像
# class Light:
#     def __init__(self,pos,rc,rr,id):
#         self.color = rc
#         self.radius = rr
#         self.pos = pos
#         self.id = id

#     def draw(self):
#         pygame.draw.circle(screen,self.color,self.pos,self.radius)


# 追光机器人
class Robot(object):
    # 机器人属性
    def __init__(self, pos=[10.0, 10.0], velocity=[0, 0]):
        self.imageSrc = pygame.image.load("50.png")
        self.rect = self.imageSrc.get_rect()
        self.image = self.imageSrc
        self.velocity = vec(velocity)
        self.vl = 0         # 初始化左轮速度
        self.vr = 0         # 初始化右轮速度
        self.angle = 0      # 初始化角度 竖直向上为0度
        self.pos = np.array(pos)
        self.rect = self.rect.move(pos[0], pos[1])  # 初始化位置
        self.step = 1
    # 当前运动方向(单位向量)
    def direction(self):
        vel = np.linalg.norm(self.velocity)
        return self.velocity/vel

    # 视野
    def vision(self,light):
        # 中心坐标
        self.pos = self.rect.center
        # 左眼坐标
        x1 = self.pos[0] + D*np.cos((ANGLE + 180 - self.angle)*np.pi/180)/2
        y1 = self.pos[1] + D*np.sin((ANGLE + 180 - self.angle)*np.pi/180)/2
        # 右眼坐标
        x2 = self.pos[0] + D*np.cos((-ANGLE - self.angle)*np.pi/180)/2
        y2 = self.pos[1] + D*np.sin((-ANGLE - self.angle)*np.pi/180)/2
        # 中心到左/右眼方向(单位向量) 
        l =  np.array([x1,y1]-np.array(self.pos))
        r =  np.array([x2,y2]-np.array(self.pos))
        norm_l = l/np.linalg.norm(l)
        norm_r = r/np.linalg.norm(r)
        # 中心到光源方向(单位向量) 
        arr_light = np.array(np.array(light.pos)-np.array(self.pos))
        norm_light = arr_light/np.linalg.norm(arr_light)
        # 分别求出  中心到光源方向 与 中心到左/右眼方向 向量夹角(角度制) arccos(a*b/(|a|*|b|))
        l_n_light = degrees(np.arccos( np.dot( norm_l,norm_light)/( np.linalg.norm(norm_l)*np.linalg.norm(norm_light)) ) )
        r_n_light = degrees(np.arccos(np.dot(norm_r,norm_light)/(np.linalg.norm(norm_r)*np.linalg.norm(norm_light))))
        # 如果向量间的夹角均小于视野角度 说明在视野范围内
        l_and_r = l_n_light + r_n_light
        if l_and_r>=VISION_ANGLE-5 and l_and_r<=VISION_ANGLE+5:
            return True
        else:
            return False
        # return (l_n_light <= VISION_ANGLE and r_n_light <= VISION_ANGLE)



    def move(self,lights):

        # 左/右眼坐标
        x1 = self.pos[0] + D*np.cos((ANGLE + 180 - self.angle)*np.pi/180)/2
        y1 = self.pos[1] + D*np.sin((ANGLE + 180 - self.angle)*np.pi/180)/2
        x2 = self.pos[0] + D*np.cos((-ANGLE - self.angle)*np.pi/180)/2
        y2 = self.pos[1] + D*np.sin((-ANGLE - self.angle)*np.pi/180)/2
       
        
        # 对于多个光源
        for light in lights: 
            self.pos = self.rect.center
            arr_light = np.array(np.array(light.pos)-np.array(self.pos))
            # 机器人中心到光源距离
            distance = np.linalg.norm(arr_light)
            # 在视野范围内 未到达光源
            if(self.vision(light) and distance >= 35):
                arrSensor = np.array([x2-x1,y2-y1])
                arr1 =  np.mat([light.pos[0] - x1,light.pos[1] - y1])
                arr2 =  np.mat([light.pos[0] - x2,light.pos[1] - y2])
                # print(arrSensor.shape,arr1.shape,arr2.shape)
                sl = float(arrSensor*arr1.T)/(np.linalg.norm(arrSensor)*np.linalg.norm(arr1))
                sr = float(arrSensor*arr2.T)/(np.linalg.norm(arrSensor)*np.linalg.norm(arr2))
                # angle_l = degrees(acos(sl))
                # angle_r = degrees(acos(sr))
                # angles.append(str(sl)+" "+str(sr))
                # angles.append(str(angle_l)+" "+str(angle_r))

                # based on rules
                self.vl = (VMAX * 0.5 * (1-sl))
                self.vr = (VMAX * 0.5 * (1+sr))
                # print(np.matrix([sl,sr]))
            
                print(self.vl,self.vr)
                self.angle -= (self.vr - self.vl)/D
                tmp = (self.vr - self.vl)/D
                dir = vec(self.direction()[0],self.direction()[1]).rotate(tmp)
                self.velocity = (self.vl + self.vr)/2 * dir
                self.rect = self.rect.move(self.velocity[0]*self.step, self.velocity[1]*self.step)
                    
            elif (self.vision(light) and distance < 35):
                x = random.randrange(30, WINDOW_WIDTH - 30)
                y = random.randrange(30, WINDOW_HEIGHT - 30)
                light_pos = np.array([x,y])
                light.pos = light_pos
            # 不在视野范围内 
            else:
                self.vl = 0
                self.vr = VMAX
                
                self.angle -= (self.vr - self.vl)/D
                # self.angle -= 1
                ag = (self.vr - self.vl)/D
                dir = vec(self.direction()[0],self.direction()[1]).rotate(ag)
                # print("out of vision!")
        
                self.velocity = (self.vl + self.vr)/2 * dir
                # print("", dir)

                self.rect = self.rect.move(self.velocity[0]*self.step, self.velocity[1]*self.step)

        # velocities.append(str(self.velocity[0])+" "+str(self.velocity[1]))

    def draw(self):
        screen.blit(self.image, self.rect)

    # 机器人中心到眼睛向量
    def draw_vectors(self):
        self.pos = self.rect.center
        x1 = self.pos[0] + D*np.cos((ANGLE + 180 - self.angle)*np.pi/180)/2
        y1 = self.pos[1] + D*np.sin((ANGLE + 180 - self.angle)*np.pi/180)/2

        x2 = self.pos[0] + D*np.cos((-ANGLE - self.angle)*np.pi/180)/2
        y2 = self.pos[1] + D*np.sin((-ANGLE - self.angle)*np.pi/180)/2
        scale = 100
        l =  np.array([x1,y1]-np.array(self.pos))
        r =  np.array([x2,y2]-np.array(self.pos))
        norm_l = l/np.linalg.norm(l)
        norm_r = r/np.linalg.norm(r)
        # left
        pygame.draw.line(screen, BLACK, self.pos, [x1,y1] + norm_l*scale, 5)
        # right
        pygame.draw.line(screen, RED, self.pos, [x2,y2] + norm_r*scale, 5)
      
    # 小车旋转
    def rotate(self):
        self.image = pygame.transform.rotate(self.imageSrc, self.angle)
        if math.fabs(self.angle) >= 360:
            self.angle -= 360
        
        self.rect = self.image.get_rect(center = self.rect.center)  # 中心矫正


def init():
    for i in range(0, 1):
        x = random.randrange(50, WINDOW_WIDTH - 50)
        y = random.randrange(50, WINDOW_HEIGHT - 50)
        robot_pos = np.array([x,y])
        # y_speed = random.randrange(-10, -5)
        y_speed = -VMAX
        robot_velocity = vec([0,y_speed])
        robot = Robot(robot_pos,robot_velocity)
        robots.append(robot)

    for j in range(0,1):
        x = random.randrange(30, WINDOW_WIDTH - 30)
        y = random.randrange(30, WINDOW_HEIGHT - 30)
        light_pos = np.array([x,y])
        light = Light(light_pos,"light.png",j) 
        lights.append(light)


 # Save file
def save_file():
    with open("input.txt","a+",encoding="utf-8") as f:
        for i in angles:
            f.writelines(i+'\n')
  
    with open("output.txt","a+",encoding="utf-8") as f2:
        for i in velocities:
            f2.writelines(i+'\n')
    f.close()
    f2.close()

clock = pygame.time.Clock()
init()
# light = Light([300,100],RED,10) 
index = 0
done = False
while not done:

    index+=1
    if mp4 == True:
        filename = 'animation/'+'capture_'+str(index)+'.jpeg'
        pygame.image.save(screen, filename)

    if index >= RUNNING_TIME:  # 控制运行时间
        done = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    x = random.randrange(30, WINDOW_WIDTH - 30)
                    y = random.randrange(30, WINDOW_HEIGHT - 30)
                    light_pos = np.array([x,y])
                    light = Light(light_pos,RED,10)
                    # light = Light(light_pos) 
              
    screen.fill(BACKGROUND_COLOR)

    # while tracking
    # light = Light(pygame.mouse.get_pos(),RED,20)
    # lights = [light]
    for light in lights:
        light.draw()
    # 将旋转后的图象，渲染到新矩形里
    for item in robots:
        item.rotate()
        item.move(lights)
        item.draw()
        item.draw_vectors()
    
    # myfont = pygame.font.SysFont("arial",20)
    # text_dist = myfont.render("Dist = "+str(distance), True, (0,255,0))
    # screen.blit(text_dist, (WINDOW_WIDTH-150, 0))
    pygame.display.update()
    # 控制帧数<=100
    clock.tick(60)

# save_file()