import cv2
import math
import numpy as np
from PIL import Image
import pyblur
import random


class raindrop():
    def __init__(self, key, centerxy=None, radius=None, input_alpha=None, input_label=None, type='default'):
        if input_label is None:
            self.key = key        
            self.ifcol = False    
            self.col_with = []
            self.center = centerxy
            self.radius = radius
            self.type = type
            # label map's WxH = 4*R , 5*R
            self.labelmap = np.zeros((self.radius * 5, self.radius * 4))
            self.alphamap = np.zeros((self.radius * 5, self.radius * 4))
            self.background = None
            self.texture = None
            self._create_label()
            self.use_label = False
        else:
            self.key = key
            assert input_alpha is not None, "Please also input the alpha map"
            self.alphamap = input_alpha
            self.labelmap = input_label
            self.ifcol = False
            self.col_with = []
            # default shape should be [h, w]            
            h, w = self.labelmap.shape
            # set the label center
            self.center = centerxy
            self.radius = min(w // 4, h // 4)
            self.background = None
            self.texture = None
            self.use_label = True

    def setCollision(self, col, col_with):
        self.ifcol = col
        self.col_with = col_with

    def updateTexture(self, bg):        
        fg = pyblur.GaussianBlur(Image.fromarray(np.uint8(bg)), 5)
        fg = np.asarray(fg)
        # add fish eye effect to simulate the background
        K = np.array([[30 * self.radius, 0, 2 * self.radius],
                    [0., 20 * self.radius, 3 * self.radius],
                    [0., 0., 1]])
        D = np.array([0.0, 0.0, 0.0, 0.0])
        Knew = K.copy()
        Knew[(0, 1), (0, 1)] = math.pow(self.radius, 1 / 3) * 2 * Knew[(0, 1), (0, 1)]
        fisheye = cv2.fisheye.undistortImage(fg, K, D=D, Knew=Knew)
        
        tmp = np.expand_dims(self.alphamap, axis=-1)      
        fisheye_resized = cv2.resize(fisheye, (tmp.shape[1], tmp.shape[0]))  
        tmp = np.concatenate((fisheye_resized, tmp), axis=2)
        
        self.texture = Image.fromarray(tmp.astype('uint8'), 'RGBA')
        # most background in drop is flipped
        self.texture = self.texture.transpose(Image.FLIP_TOP_BOTTOM)

    # create the raindrop label
    def _create_label(self):
        if self.type == "default":
            self._createDefaultDrop()
        elif self.type == "splash":
            self._createSplashDrop()

    def _createDefaultDrop(self):
        cv2.circle(self.labelmap, (self.radius * 2, self.radius * 3), self.radius, 128, -1)
        cv2.ellipse(self.labelmap, (self.radius * 2, self.radius * 3), (self.radius, int(1.3 * math.sqrt(3) * self.radius)), 0, 180, 360, 128, -1)        
        # set alpha map for png 
        self.alphamap = pyblur.GaussianBlur(Image.fromarray(np.uint8(self.labelmap)), 10)
        self.alphamap = np.asarray(self.alphamap).astype(np.float64)
        self.alphamap = self.alphamap / np.max(self.alphamap) * 255.0
        # set label map
        self.labelmap[self.labelmap > 0] = 1

    def _createSplashDrop(self):
        num_points = random.randint(12, 20)  # Daha fazla noktayla daha karmaşık bir şekil
        splash_points = []
        
        # Şekil için rastgele noktalar oluşturuluyor
        for i in range(num_points):
            angle = math.radians(i * (360 / num_points))
            radius = self.radius + random.randint(-60, 60)  # Düzensiz kenarlar
            x = int(self.radius * 2 + math.cos(angle) * radius)
            y = int(self.radius * 3 + math.sin(angle) * radius)
            # Koordinatların belirlenen sınırlar içinde kalmasını sağla
            x = max(self.radius, min(x, self.labelmap.shape[1] - self.radius - 1))
            y = max(self.radius, min(y, self.labelmap.shape[0] - self.radius - 1))
            splash_points.append((x, y))
        
        # Bu rastgele noktaları birleştirip şekli oluşturuyoruz
        splash_points = np.array(splash_points, dtype=np.int32)
        
        # Şekli labelmap'e çiziyoruz
        cv2.polylines(self.labelmap, [splash_points], isClosed=True, color=128, thickness=1)
        cv2.fillPoly(self.labelmap, [splash_points], color=128)
        
        # Alpha haritası oluşturuluyor (şekil transparanlık için)
        self.alphamap = pyblur.GaussianBlur(Image.fromarray(np.uint8(self.labelmap)), 10)
        self.alphamap = np.asarray(self.alphamap).astype(np.float64)
        self.alphamap = self.alphamap / np.max(self.alphamap) * 255.0
        self.labelmap[self.labelmap > 0] = 1
    
    def setKey(self, key):
        self.key = key

    def getLabelMap(self):
        return self.labelmap

    def getAlphaMap(self):
        return self.alphamap

    def getTexture(self):
        return self.texture

    def getCenters(self):
        return self.center
        
    def getRadius(self):
        return self.radius

    def getKey(self):
        return self.key

    def getIfColli(self):
        return self.ifcol

    def getCollisionList(self):
        return self.col_with
    
    def getUseLabel(self):
        return self.use_label
