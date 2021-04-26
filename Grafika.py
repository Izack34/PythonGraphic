#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
ma=np.ones((10,10,4))
#print(ma)


# In[15]:


import matplotlib.pyplot as plt
import numpy as np

ma=np.ones((10,10,4))

#print(ma)
 
plt.imshow(ma)


# In[18]:


ma[4][5]=[0,0,0,1]
ma[4][4]=[0,1,0,1]
ma[4][3]=[1,0,0,1]
ma[4][2]=[0,0,1,1]
plt.imshow(ma)


# In[4]:


ma=np.ones((20,20,4))
def prostokat(c,color):
    """rysowanie prodtokata po kordach"""
    for x in range(c[0][0],c[1][0]+1):
        ma[c[0][1]][x]=color
    for x in range(c[0][1],c[1][1]):
        ma[x][c[0][0]]=color
    for x in range(c[0][0],c[1][0]+1):
        ma[c[1][1]][x]=color
    for x in range(c[0][1],c[1][1]):
        ma[x][c[1][0]]=color
    


# In[5]:


prostokat(((2,2),(13,14)),[0,0,0,1])
plt.imshow(ma)


# In[ ]:





# In[8]:


def get_line(start, end, color):

    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
 
    is_steep = abs(dy) > abs(dx)
 

    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 

    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
 

    dx = x2 - x1
    dy = y2 - y1
 

    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
 
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
 
    if swapped:
        points.reverse()
    for x in points:
        ma[x[1]][x[0]]=color
        
        
ma=np.ones((20,20,4))
get_line([2,18],[15,3],[0,0,0,1])
plt.imshow(ma)


# In[42]:


ma=np.ones((20,20,4))
def trójkąt(punkty):
    get_line(punkty[0],punkty[1],[0,0,0,1])
    get_line(punkty[1],punkty[2],[0,0,0,1])
    get_line(punkty[0],punkty[2],[0,0,0,1])
    
trójkąt([[18,18],[15,3],[3,14]])
plt.imshow(ma)


# In[43]:



ma=np.ones((20,20,4))


def floodfill(x, y, oldColor, newColor):

    theStack = [(x, y)]

    while len(theStack) > 0:

        x, y = theStack.pop()
        check=False
        for i in range(0,3): 
            if ma[x][y][i] != oldColor[i]:
                check=True
        if check:
            continue

        ma[x][y] = newColor

        theStack.append( (x + 1, y) )  # right

        theStack.append( (x - 1, y) )  # left

        theStack.append( (x, y + 1) )  # down

        theStack.append( (x, y - 1) )  # up

ma=np.ones((20,20,4))
#prostokat(((2,2),(13,14)),[0,0,0,1])
trójkąt([[18,18],[15,3],[3,14]])
floodfill(14,13,[1,1,1,1],[0,1,1,1])
#print(ma)
plt.imshow(ma)


# In[9]:


for x in range(0,3):
    print("f")


# In[48]:


ma=np.ones((10,10,4))
def midPointCircleDraw(x_centre,  
                       y_centre, r): 
  
    x = r 
    y = 0
      
    ma[y + y_centre][x + x_centre] = [0,0,0,1]
          
    if (r > 0) : 
        ma[-y + y_centre][-x + x_centre] = [0,0,0,1]
        
        ma[x + y_centre][y + x_centre] = [0,0,0,1]
        
        ma[-x + y_centre][-y + x_centre] = [0,0,0,1]
        
      
    
    P = 1 - r  
    while (x > y) : 
      
        y += 1
                  
        if (P <= 0):  
            P = P + 2 * y + 1
                       
        else:          
            x -= 1
            P = P + 2 * y - 2 * x + 1
                  
        if (x < y): 
            break
                      
        ma[y + y_centre][x + x_centre] = [0,0,0,1]
        ma[y + y_centre][-x + x_centre] = [0,0,0,1]
        ma[-y + y_centre][x + x_centre] = [0,0,0,1]
        ma[-y + y_centre][-x + x_centre] = [0,0,0,1]
                  
        if (x != y) : 

            ma[x + y_centre][y + x_centre] = [0,0,0,1]
            ma[x + y_centre][-y + x_centre] = [0,0,0,1]
            ma[-x + y_centre][y + x_centre] = [0,0,0,1]
            ma[-x + y_centre][-y + x_centre] = [0,0,0,1] 
            
                              

      
     
midPointCircleDraw(3, 3, 1) 
  
plt.imshow(ma)


# In[ ]:





# In[45]:


#pokemon

ma=np.ones((25,30,4))
 

get_line((11,14),(15,8),[0,0,0,1])#body
get_line((15,7),(18,4),[0,0,0,1])
get_line((18,4),(21,4),[0,0,0,1])
get_line((18,4),(21,4),[0,0,0,1])
get_line((18,4),(21,4),[0,0,0,1])
get_line((21,4),(25,8),[0,0,0,1])
get_line((25,8),(25,10),[0,0,0,1])
get_line((25,10),(20,13),[0,0,0,1])
get_line((20,13),(19,13),[0,0,0,1])
get_line((20,13),(20,15),[0,0,0,1])
get_line((21,17),(19,18),[0,0,0,1])
get_line((17,19),(20,16),[0,0,0,1])
get_line((16,19),(16,20),[0,0,0,1])
get_line((12,19),(13,19),[0,0,0,1])
get_line((15,19),(16,19),[0,0,0,1])
get_line((16,21),(12,21),[0,0,0,1])
get_line((12,21),(12,18),[0,0,0,1])
get_line((12,18),(11,18),[0,0,0,1])#koniec

get_line((11,18),(11,13),[0,0,0,1])
get_line((11,13),(8,11),[0,0,0,1])#ogon
get_line((8,11),(8,10),[0,0,0,1])
get_line((9,9),(9,7),[0,0,0,1])
get_line((9,7),(7,3),[0,0,0,1])
get_line((7,3),(5,5),[0,0,0,1])
get_line((5,5),(5,6),[0,0,0,1])
get_line((4,7),(4,9),[0,0,0,1])
get_line((5,10),(6,10),[0,0,0,1])
get_line((6,10),(6,12),[0,0,0,1])
get_line((7,13),(7,14),[0,0,0,1])
get_line((7,14),(11,18),[0,0,0,1])#koniec ogonu

#reka
get_line((15,14),(17,13),[0,0,0,1])
get_line((16,12),(16,12),[0,0,0,1])
#k
#yellow [0.96,0.96,0,1]
get_line((7,11),(6,8),[0.96,0.96,0,1])


floodfill(16,10,[1,1,1,1],[0.94,0.6,0.17,1])

floodfill(17,15,[1,1,1,1],[0.94,0.6,0.17,1])

floodfill(7,7,[1,1,1,1],[0.98,0,0,1])

#orange [0.94,0.6,0.17,1]

#oko
prostokat(((19,7),(20,9)),[0,0,0,1])
get_line((20,7),(20,7),[1,1,1,1])

#brzuch
get_line((16,18),(16,15),[0.96,0.96,0,1])
get_line((18,13),(18,13),[0.96,0.96,0,1])
floodfill(15,17,[0.94,0.6,0.17,1],[0.96,0.96,0,1])



plt.imshow(ma)


# In[49]:


ma=np.ones((25,30,4))
 
get_line((11,14),(15,8),[0,0,0,1])#body
get_line((15,7),(18,4),[0,0,0,1])
get_line((18,4),(21,4),[0,0,0,1])
get_line((18,4),(21,4),[0,0,0,1])
get_line((18,4),(21,4),[0,0,0,1])
get_line((21,4),(25,8),[0,0,0,1])
get_line((25,8),(25,10),[0,0,0,1])
get_line((25,10),(20,13),[0,0,0,1])
get_line((20,13),(19,13),[0,0,0,1])
get_line((20,13),(20,15),[0,0,0,1])
get_line((21,17),(19,18),[0,0,0,1])
get_line((17,19),(20,16),[0,0,0,1])
get_line((16,19),(16,20),[0,0,0,1])
get_line((12,19),(13,19),[0,0,0,1])
get_line((15,19),(16,19),[0,0,0,1])
get_line((16,21),(12,21),[0,0,0,1])
get_line((12,21),(12,18),[0,0,0,1])
get_line((12,18),(11,18),[0,0,0,1])#koniec
get_line((11,13),(8,11),[0,0,0,1])#ogon
get_line((8,11),(8,10),[0,0,0,1])
get_line((9,9),(9,7),[0,0,0,1])
get_line((9,7),(7,3),[0,0,0,1])
get_line((7,3),(5,5),[0,0,0,1])
get_line((5,5),(5,6),[0,0,0,1])
get_line((4,7),(4,9),[0,0,0,1])
get_line((5,10),(6,10),[0,0,0,1])
get_line((6,10),(6,12),[0,0,0,1])
get_line((7,13),(7,14),[0,0,0,1])
get_line((7,14),(11,18),[0,0,0,1])#koniec ogonu
plt.imshow(ma)


# In[39]:


get_line((11,13),(8,11),[0,0,0,1])#ogon
get_line((8,11),(8,10),[0,0,0,1])
get_line((9,9),(9,7),[0,0,0,1])
get_line((9,7),(7,3),[0,0,0,1])
get_line((7,3),(5,5),[0,0,0,1])
get_line((5,5),(5,6),[0,0,0,1])
get_line((4,7),(4,9),[0,0,0,1])
get_line((5,10),(6,10),[0,0,0,1])
get_line((6,10),(6,12),[0,0,0,1])
get_line((7,13),(7,14),[0,0,0,1])
get_line((7,14),(11,18),[0,0,0,1])#koniec ogonu
plt.imshow(ma)


# In[46]:


#reka
get_line((15,14),(17,13),[0,0,0,1])
get_line((16,12),(16,12),[0,0,0,1])
#k
#yellow [0.96,0.96,0,1]
get_line((7,11),(6,8),[0.96,0.96,0,1])


floodfill(16,10,[1,1,1,1],[0.94,0.6,0.17,1])

floodfill(17,15,[1,1,1,1],[0.94,0.6,0.17,1])

floodfill(7,7,[1,1,1,1],[0.98,0,0,1])

#orange [0.94,0.6,0.17,1]

#oko
prostokat(((19,7),(20,9)),[0,0,0,1])
get_line((20,7),(20,7),[1,1,1,1])

#brzuch
get_line((16,18),(16,15),[0.96,0.96,0,1])
get_line((18,13),(18,13),[0.96,0.96,0,1])
floodfill(15,17,[0.94,0.6,0.17,1],[0.96,0.96,0,1])
plt.imshow(ma)


# In[ ]:





# In[115]:


#pokemon
plt.imshow(ma)


# In[ ]:





# In[2]:





def plot(x, y, c):
    #print(c)

    ma[y][x]=[1-c,1-c,1-c,1]

def ipart(x) :
    return int(x)

def roundd(x):
    return ipart(x + 0.5)

def fpart(x):
    return x - int(x)

def rfpart(x):
    return 1 - fpart(x)

def lineanty(x0,y0,x1,y1):
    is_steep = abs(y1 - y0) > abs(x1 - x0)
    
    if is_steep:
        x1, y1 = y1, x1
        x0, y0 = y0, x0
        
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    
    dx = x1 - x0
    dy = y1 - y0
    gradient = dy / dx
    if dx == 0.0:
        gradient = 1.0
    
    xend = roundd(x0+1)
    yend = y0 + gradient * (xend - x0)
    xgap = rfpart(x0+0.5)
    xpxl1 = xend
    ypxl1 = ipart(yend)
    
    if is_steep :
        plot(ypxl1, xpxl1, rfpart(yend) * xgap)
        plot(ypxl1+1, xpxl1,  fpart(xend) * xgap)
    else:
        plot(xpxl1, ypxl1, rfpart(yend) * xgap)
        plot(xpxl1, ypxl1+1,  fpart(xend) * xgap)

    intery = yend + gradient
    
    xend = roundd(x1+1)
    yend = y1 + gradient * (xend - x1)
    xgap = fpart(x1 +0.5)
    xpxl2 = xend 
    ypxl2 = ipart(yend)
    
    if is_steep:
        plot(ypxl2  , xpxl2, rfpart(xend) * xgap)
        plot(ypxl2+1, xpxl2,  fpart(yend) * xgap)
    else:
        plot(xpxl2, ypxl2,  rfpart(xend) * xgap)
        plot(xpxl2, ypxl2+1, fpart(yend) * xgap)
    
    
    if is_steep:
        for x in range(xpxl1 + 1,xpxl2):
                plot(ipart(intery)  , x,rfpart(intery))
                plot(ipart(intery)+1, x,fpart(intery))
                intery = intery + gradient
    else:
        for x in range(xpxl1 + 1, xpxl2):
                plot(x, ipart(intery),  rfpart(intery))
                plot(x, ipart(intery)+1,fpart(intery))
                intery = intery + gradient
           


# In[3]:


ma=np.ones((40,40,4))


# In[4]:


lineanty(10,10,33,27)
lineanty(10,10,38,27)
plt.imshow(ma)


# In[104]:





# In[ ]:




