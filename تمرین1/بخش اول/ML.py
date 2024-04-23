import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

#قسمت الف : فراخوانی مجموعه داده + نمایش تعداد و ابعاد داده های آموزش و آزمون 
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


print("تعداد داده‌های آموزش:", len(x_train) , "\n")
print("ابعاد داده‌های آموزش:", x_train.shape , "یعنی 60000 تصویر با ابعاد 28*28 پیکسل\n")
print("ابعاد برچسب‌های آموزش:", y_train.shape, "\n")

print("تعداد داده‌های آزمون:", len(x_test), "\n")
print("ابعاد داده‌های آزمون:", x_test.shape , "یعنی 10000 تصویر با ابعاد 28*28 پیکسل\n")
print("ابعاد برچسب‌های آزمون:", y_test.shape, "\n")





# قسمت ب : یک نمونه از هر کلاس را نمایش میدهیم
num_classes = 10

fig, axes = plt.subplots(nrows=1, ncols=num_classes, figsize=(15,5))

for class_label in range(num_classes):
    # پیدا کردن اولین تصویر از هر کلاس
    # (من در این جا اولین تصویر از هر کلاس را نمایش دادم میتوان تصویر دوم یا سوم و یا ... را نیز نمایش داد)
    index = np.where(y_train == class_label)[0][0]
    image = x_train[index]

    # نمایش تصویر
    axes[class_label].imshow(image, cmap='RdPu')
    axes[class_label].set_title(f'Class {class_label}')
    axes[class_label].axis('off')

plt.tight_layout()
plt.show()





#قسمت ج : نمودار هیستوگرام مربوط به تعداد نمونه های  هر کلاس برای  داده های آموزش 
class_counts = np.bincount(y_train)
colors = np.random.rand(10, 3)
plt.bar(range(10), class_counts, color=colors, tick_label=range(10))
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Histogram of Class Counts in Training Data')
plt.show()





#قسمت د : نرمالسازی داده ها در بازه ی صفر و یک 
min_value = np.min(x_train)
max_value = np.max(x_train)
# MinMax normalization formula : y = ( x − min ⁡ ) / ( max ⁡ − min ⁡ )
x_train_normalized = (x_train - min_value) / (max_value - min_value)

num_samples = 10  
plt.figure(figsize=(15, 5))  
for i in range(num_samples):
    plt.subplot(1, num_samples, i+1)
    plt.imshow(x_train_normalized[i], cmap='cool')
    plt.axis('off')
    # plt.title('Class{}'.format(i))
    
plt.suptitle('Classes after normalization', x=0.5, y=0.70, fontsize=14, ha='center' , color='#FF1493', fontweight='bold') 
plt.tight_layout(rect=[0, 0, 0.98, 0.9])
plt.show()
# چاپ مقادیر مینیمم و ماکسیمم برای بررسی صحت اسکیل کردن
print("Min value after normalization:", np.min(x_train_normalized))
print("Max value after normalization:", np.max(x_train_normalized))





