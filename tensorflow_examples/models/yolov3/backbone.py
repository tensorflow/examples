import tensorflow as tf
import commonBlocks as com

def darknet53(input_data):
    input_data=com.convolutional(input_data,(3,3,3,32))
    input_data=com.convolutional(input_data,(3,3,32,64),downsample=True)

    for i in range(1):
        input_data=com.residual_block(input_data,64,32,64)
    
    input_data=com.convolutional(input_data,(3,3,64,128),downsample=True)

    for i in range(2):
        input_data=com.residual_block(input_data,128,64,128)

    input_data=com.convolutional(input_data,(3,3,128,256),downsample=True)
    
    for i in range(9):
        input_data=com.residual_block(input_data,256,128,256)

    route1=input_data
    input_data= com.convolutional(input_data,(3,3,256,512),downsample=True)

    for i in range(8):
        input_data=com.residual_block(input_data,512,256,512)
    
    route2= input_data
    input_data= com.convolutional(input_data,(3,3,512,1024),downsample=True)

    for i in range(4):
        input_data=com.residual_block(input_data,1024,512,1024)
    
    return route1,route2,input_data