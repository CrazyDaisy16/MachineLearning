import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

#تبدیلات اولیه
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Batch Size = 200
# در این جا فرض بر این است که اندازه ی  هر بچ 200 است
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False)

# تعریف مدل MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 120)   # 120 لایه ورودی به لایه پنهان اول
        self.fc2 = nn.Linear(120, 84)      # 84لایه پنهان اول به لایه پنهان دوم
        self.fc3 = nn.Linear(84, 10)       # لایه پنهان دوم به لایه خروجی
        #چون هر تصویر به یکی از 10 دسته صفر تا نه تعلق دارد خروجی شبکه ی ما 10 است
        self.relu = nn.ReLU()              # تابع فعال‌ساز ReLU

    def forward(self, x):
        x = x.view(-1, 28*28)  # تغییر شکل تصویر ورودی به بردار
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLP()

# تابع هزینه و بهینه سازی
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)



# آموزش مدل و ارزیابی آن بعد از هر ایپاک
for epoch in range(5):  #ما 10000 داده ی آزمون داریم و اندازه ی هر بچ 200 است پس ما 5 دور مدل را آموزش میدهیم
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)



        loss = criterion(outputs, labels)# محاسبه هزینه با استفاده از تابع هزینه‌ی Cross Entropy Loss
        # ما از تابع criterion برای محاسبه هزینه بر حسب به cross entropy استفاده کردیم



        loss.backward()# محاسبه گرادیان‌ها
        optimizer.step()
        # به‌روزرسانی وزن‌ها با استفاده از بهینه‌سازی مورد نظر
        #پس انتشار یا همان backpropagation و اپدیت وزن ها با دو تابع backward() ,  optimizer.step() انجام دادیم



        running_loss += loss.item()
        if i % 50 == 49:
             #وضعیت آموزش هر 50 دسته را چاپ میکنیم و در نهایت بعد از 5 دور آموزش اتمام آموزش را پرینت میکنیم
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0

    # ارزیابی
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += criterion(outputs, labels).item()
    
    # چاپ دقت و هزینه‌ی آزمون برای ایپاک جاری
    print('Epoch %d - Test Loss: %.3f, Accuracy: %.2f %%' % (
            epoch + 1, test_loss / len(testloader), 100 * correct / total))

print('Finished Training')



