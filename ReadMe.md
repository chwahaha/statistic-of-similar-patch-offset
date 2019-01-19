# Image Completion approaches using the statistics of Similar Patches"



参考了一些Github上的代码自己实现了何凯明大神的这个算法，算法思路比较简单，但是效果还不错。

第二个man的效果比paper中的效果差一些，可能是由于计算patch offset的时候使用的是PatchMatch，其精度速度都会更差一些。其他参数我都只做了一些简单的调整。



| 原图                         | label                       | result |
| ---------------------------- | --------------------------- | ------ |
| ![](.\testsImage\image4.jpg) | ![](.\Output\lableMap4.jpg) | ![](.\Output\result4.jpg)        |
| ![](.\testsImage\man.png) |     ![](.\Output\lableMap-man.jpg)                        |  ![](.\Output\result-man.jpg)      |

# Implementation detail

1. calculate the patch offsets  histogram by PatchMatch
2. get the k dominate offset  
3. multi-label by graph-cut
4. gradient-domain fusion

# Requires

OpenCV

developed by vs2017 community

tested only on win10

# reference







