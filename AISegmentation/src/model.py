import pdb
import traceback
import tensorflow as tf

class ConvBlock3D(tf.keras.layers.Layer):

    def __init__(self, filters, pool, kernel_size=(3,3,3), dropout=None, trainable=False, name=''):
        super(ConvBlock3D, self).__init__(name='{}_ConvBlock3D'.format(name))

        self.filters = filters
        self.pool = pool
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.trainable = trainable

        self.conv_layer = tf.keras.Sequential()
        for filter_id, filter_count in enumerate(self.filters):
            self.conv_layer.add(
                tf.keras.layers.Conv3D(filter_count, self.kernel_size, padding='same'
                        , kernel_regularizer=tf.keras.regularizers.l2(0.1)
                        , activation='relu'
                        , trainable = self.trainable
                        , name='Conv_{}'.format(filter_id))
            )
            self.conv_layer.add(tf.keras.layers.BatchNormalization(trainable=self.trainable))
            if filter_id == 0 and self.dropout is not None:
                self.conv_layer.add(tf.keras.layers.Dropout(rate=self.dropout, name=self.name + '_DropOut'))
        
        self.pool_layer = tf.keras.layers.MaxPooling3D((2,2,2), strides=(2,2,2), trainable=self.trainable, name='Pool_{}'.format(self.name))
    
    def call(self, x):
        
        x = self.conv_layer(x)
        
        if self.pool is False:
            return x
        else:
            x_pool = self.pool_layer(x)
            return x, x_pool

class UpConvBlock3D(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size=(2,2,2), strides=(2, 2, 2), trainable=False, name=''):
        super(UpConvBlock3D, self).__init__(name='{}_UpConv3D_Concat'.format(name))
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.trainable = trainable

        self.upconv = tf.keras.layers.Conv3DTranspose(self.filters, self.kernel_size
                        , strides=self.strides
                        , kernel_regularizer=tf.keras.regularizers.l2(0.1)
                        , trainable=self.trainable
                        , name='UpConv_{}'.format(self.name))
        
    def call(self, x1, x2):
        x1_upconv = self.upconv(x1)
        return tf.concat([x1_upconv, x2], axis=-1, name='Concat_{}'.format(self.name))

class ModelUNet3D(tf.keras.Model):

    def __init__(self, class_count, activation='softmax', trainable=False, verbose=False):
        super(ModelUNet3D, self).__init__()

        self.class_count = class_count
        self.verbose = verbose
        self.activation = activation #activation=[None, softmax, sigmoid]
        self.trainable = trainable

        if 1:
            self.convblock1 = ConvBlock3D(filters=[8, 8]    , pool=True , dropout=0.1, trainable=self.trainable, name='Block1')
            self.convblock2 = ConvBlock3D(filters=[16, 16]  , pool=True , dropout=0.1, trainable=self.trainable, name='Block2')
            self.convblock3 = ConvBlock3D(filters=[32, 32]  , pool=True , dropout=0.1, trainable=self.trainable, name='Block3')
            self.convblock4 = ConvBlock3D(filters=[64, 64]  , pool=True , dropout=0.2, trainable=self.trainable, name='Block4')
            self.convblock5 = ConvBlock3D(filters=[128, 128], pool=False, dropout=0.3, trainable=self.trainable, name='Block5')

            self.upconvblock6 = UpConvBlock3D(filters=64, trainable=self.trainable, name='Block6_1')
            self.convblock6   = ConvBlock3D(filters=[64,64], pool=False, dropout=0.2, trainable=self.trainable, name='Block6_2') 
            self.upconvblock7 = UpConvBlock3D(filters=32, trainable=self.trainable, name='Block7_1')
            self.convblock7   = ConvBlock3D(filters=[32,32], pool=False, dropout=0.1, trainable=self.trainable, name='Block7_2')
            self.upconvblock8 = UpConvBlock3D(filters=16, trainable=self.trainable, name='Block8_1')
            self.convblock8   = ConvBlock3D(filters=[16,16], pool=False, dropout=0.1, trainable=self.trainable, name='Block8_2')
            self.upconvblock9 = UpConvBlock3D(filters=8, trainable=self.trainable, name='Block9_1')
            self.convblock9   = ConvBlock3D(filters=[self.class_count,self.class_count], pool=False, dropout=0.1, trainable=self.trainable, name='Block9_2')
            self.convblock10  = tf.keras.layers.Conv3D(filters=self.class_count, kernel_size=(1,1,1), padding='same'
                                , activation=self.activation , trainable=self.trainable
                                , name='Block10')
    
    def call(self,x):
        try:
            # conv1, pool1 = ConvBlock(filters=[8, 8]    , pool=True , dropout=0.1, name='Block1')(x)
            conv1, pool1 = self.convblock1(x)
            conv2, pool2 = self.convblock2(pool1)
            conv3, pool3 = self.convblock3(pool2)
            conv4, pool4 = self.convblock4(pool3)
            conv5        = self.convblock5(pool4)
            
            up6   = self.upconvblock6(conv5, conv4)
            conv6 = self.convblock6(up6)
            up7   = self.upconvblock7(conv6, conv3)
            conv7 = self.convblock7(up7)
            up8   = self.upconvblock8(conv7, conv2)
            conv8 = self.convblock8(up8)
            up9   = self.upconvblock9(conv8, conv1)
            conv9 = self.convblock9(up9)
            conv10 = self.convblock10(conv9)
            
            if self.verbose:
                print (' - x:', x.shape)
                print (' - conv1: ', conv1.shape, ' || pool1: ', pool1.shape)                
                print (' - conv2: ', conv2.shape, ' || pool2: ', pool2.shape)                
                print (' - conv3: ', conv3.shape, ' || pool3: ', pool3.shape)                
                print (' - conv4: ', conv4.shape, ' || pool4: ', pool4.shape)                
                print (' - conv5: ', conv5.shape)
                print (' - conv6: ', conv6.shape)                
                print (' - conv7: ', conv7.shape)                
                print (' - conv8: ', conv8.shape)                
                print (' - conv9: ', conv9.shape)
                print (' - conv10: ', conv10.shape)
                            
            return conv10

        except:
            traceback.print_exc()
            pdb.set_trace()
