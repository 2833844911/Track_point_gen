ai轨迹模拟

把你的连续轨迹放gj.json 文件 
然后训练 启动 train.py

测试 使用 test.py

轨迹搜集把 point.html 放浏览器 然后在页面滑动 
调用 copy(JSON.stringify(points)) 保存轨迹

# 设置训练生成轨迹点的个数范围
fawei = [30, 150] # 训练30个点到150个点的轨迹生成


 ![image](test.jpg)
