# feiniao  学习过程中的代码:  
## 2020/11/29
**common**是随书的一个包，里面是成套的NN工具，包括各种function, layers, trainer  
**dataset**是随书的一个包，里面是有关mnist的库、和一些资源文件  
**未弄懂**是学长给的手写数字识别的代码，暂未完全看懂  
**epoch**是一个比较完整的实例，但是运行巨慢  
**main**和**epoch**差不多，但因调用的two_layer_net，所以快了很多  
**two_layer_net**和**two_layers_net**：没s的里面有更高效的gradient  
其他的文件如名，没有太多实际的内容，很多是固定参数，不成套的内容  
**sample_weight.pkl**是一个用于保存weight值的文件(实际上只有神经网络的推理这个文件用到了)  
## 2020/12/6
更新了**two_layer_net**，**main**  
**two_layer_net**：使用有序字典添加了生成层，使用了ReLu  
**main**：用新的**two_layer_net**，效果好多了，有loss的输出  
我很想它还能显示图像什么的，但是失败了  
**back-layers**和**layer_naive**是学反向传播写的代码
## 2020/12/13
更新了**显示图像**，**main**  
**显示图像**：多了个trans，来转化one-hot-vector   
**main**：现在可以显示图像了  
