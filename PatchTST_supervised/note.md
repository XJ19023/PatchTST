# 写作表达
1. The advent of models like ChatGPT has illuminated the new direction in AI development.
1. Our motivation `stems from`
1. Consider the example in Figure 2.
1. 双栏图片宽度11cm.
1. 图片字体Arial，字号9或10.
1. 图表题字体Times New Roman，子图字号8.

---
---

# 写作引用
1. transformers are more sensitive than CNNs to quantization error.
1. a rightshifting sequential multiplier as it requires a smaller firststage adder than a left-shifting design, preventing long carry propagation and sign-bit extension.

---
---
`
# 软件使用
># Zotero
>1. `+/-` 展开，折叠所有条目

># Linux
>1. `lsblk` 查看已挂载设备
>1. `umount /dev/sdX` 卸载U盘
>1. `eject  /dev/sdX` 弹出U盘
>1. `sed -i 's/old_string/new_string/g' *.txt` 批量替换字符串
>1. `find . -name "epoch*" | xargs du -sh | sort -hr` 降序查看特定文件大小
>1. `find . -name "*.log" | xargs sed -i '/Running tokenizer on dataset/d'` 删除特定行
>1. 远程后台
```bash
#!/bin/bash
nohup ./run_all.sh > output.log 2>&1 &
```
>1. 查看gpu进程
```bash
apt-get install  psmisc
fuser -v /dev/nvidia*
```

- 远程服务器使用本地代理
```bash
# powershell
ssh -vvv -N -R 7897:localhost:7897 -p 30140 root@10.210.22.111 # A10
ssh -vvv -N -R 7897:localhost:7897 -p 32041 root@10.210.22.35 # A40
ssh -vvv -N -R 7897:localhost:7897 -p 30421 root@10.210.22.137 # A100
# 服务器
export http_proxy=http://127.0.0.1:7897;  #HTTP
export https_proxy=http://127.0.0.1:7897; #HTTPS

curl https://scholar.google.com # test

unset http_proxy  #HTTP
unset https_proxy #HTTPS
```
<div align=center> 
  <img src="FigureForNote/clash.png" width="500px" />
</div>

>1. scp window-wsl端
- 将window下的.ssh.id_rsa保存到wsl并修改权限 `chmod 600 id_rsa`
- `scp -i id_rsa <local_file> user_name@host_addr:<target path>`
- `scp -i 'C:\Users\11833\.ssh\id_rsa' -P 30140 -r  local_file root@10.210.22.111:`
- `scp -i 'C:\Users\11833\.ssh\id_rsa' -P 6000 -r juxin@119.29.236.16:~/smartBlockFig.tar.gz .`
- `scp -i 'C:\Users\11833\.ssh\id_rsa' -P 30426 -r root@10.210.22.113:/cephfs/shared/juxin/dataset/val.jsonl.zst .`
- `scp -P 10003 source/rtl/*.v juxin@10.220.7.250:~/qwt/dc/rtl` 

>6. conda环境跨用户复制('lm-eval' from hejun to juxin)
>6. `pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple` pip临时换源
>6. `pip install --upgrade bitsandbytes==0.43.0` pip更新软件
```bash
cd ~/.conda/envs/lm-eval
find . -type f -name "*" -exec sed -i 's/hejun/juxin/g' {} +
```
>6. `lm-eval`和`bert_mx`都是从何俊那直接复制的conda环境
>6. python版本切换及pip对应
```bash
apt install python3.9
cd /usr/bin $$ ln -s python3.9 python
wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py
pip -V
```
1. pip 换源和重置
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config unset global.index-url
```


># Excel
>1. `=C6/$C$10` 固定C10

># VSCode
>1. `Ctrl D` 选中光标所在的下一个单词
>1. `Ctrl `` 打开终端
>1. `Alt Up/Down` 上/下移动代码行
>1. `Alt Shift Up/Down` 上/下复制代码行
>1. `Alt 鼠标左键` 多个光标
>1. `Ctrl K, Ctrl 1 ` 折叠一层
>1. `Ctrl K J` 展开所有代码
>1. `F11` 切换全屏
># vscode-ipynb
>1. `ctrl shif -` 拆分单元格 

># Anaconda
>1. `conda -V` 查看conda版本
>1. `conda env list` 查看conda虚拟环境
>1. `conda install package_name -n env_name` 在env_name安装包
>1. `conda uninstall/remove package_name` 在当前虚拟环境卸载包
>1. `conda create -n env_name python=3.8` 创建虚拟环境
>1. `conda remove -n env_name --all` 删除虚拟环境
>1. `conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia` 安装pytorch
>1. `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia` 安装pytorch
>1. `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` 安装pytorch
>1. `./cuda_11.7.0_515.43.04_linux.run --installpath=/root/cuda` 安装cuda
>1. `conda update --all` 遇到问题先试试这个
---
---
># GIT
``` bash
ssh-keygen -t ed25519 -C "1183300812@qq.com" #生成密钥
git remote -v # 查看当前ssh地址
git remote set-url origin git@github.com:XJ19023/QwT-LLM.git # 改用ssh地址
eval "$(ssh-agent -s)"
ssh-add /cephfs/juxin/.ssh/id_rsa
ssh -T git@github.com # 查看是否有权限，没有的话则添加权限
ssh -vT git@github.com # 调试模式 输出log
```
>1. `git tag -a v1.0 -m "bianli alpha" HEAD` 创建标签 
>1. `git push origin v2.0` 推送标签到远程
>1. `git pull --rebase origin main` 将远程仓库的内容拉取到本地并合并,避免冲突
>1. `git pull --rebase origin main` 将远程仓库的内容拉取到本地并合并,避免冲突
>1. `git commit -m "add spark ppl"` 提交代码
>1. `git push` 将本地仓库的内容推送到远程仓库
>1. `git restore readme.md` 撤销本地修改
>1. `git restore .` 撤销所有本地修改
>1. `git restore --staged <file>` 撤销 git add .
>1. `git reset --mixed HEAD~1` 撤销 git commit到未add状态
>1. `git diff --name-only | xargs -I {} cp {} /cephfs/juxin/gitmodify` 复制修改的文件到指定目录
>1. 下载大文件
```bash
apt install git-lfs
mkdir tmp && cd tmp
git init
git lfs install
git clone ...
```


---
---
># Vim
>1. `/\<set\>\C` 正向搜索 `set` 
>1. `?\<set\>` 反向搜索 `set`
>1. `*` 正向搜索光标所在单词
>1. `#` 反向搜索光标所在单词
>1. `:g/set/#` 显示所有搜索结果，`/#`显示行号
>1. `:g/^\s*字段名\s*$/d` 删除与字段名匹配的行，`\s*`匹配多个空格
>1. `:reg` 查看剪贴板
>1. `“Ny` 复制到`N`号剪贴板
- .vimrc
```bash
set number
inoremap jk <Esc>
inoremap ( ()<ESC>i
inoremap [ []<ESC>i
inoremap { {}<ESC>i
inoremap " ""<ESC>i
inoremap ' ''<ESC>i
```

---
---
># tmux
>1. `Ctrl+b “` 划分为上下窗格 
>1. `Ctrl+b %` 划分为左右窗格 
>1. `Ctrl+b z` 平铺当前窗格 
>1. `Ctrl+b n` 切换窗格 
>1. `Ctrl+b Ctrl+ <arrow>` 调整窗格大小 
>1. `Ctrl+b c` 创建新窗口
>1. `Ctrl+b <arrow>` 选择窗格
>1. `Ctrl+b $` 会话重命名
>1. `Ctrl+b s` 查看所有会话
>1. `Ctrl+b t` 显示时钟
>1. `Ctrl+b f` 查找文本

---
---
# Python pdb
1. breakpoint()  or python -m pdb ***.py
1. `n`, next
1. `s`, step
1. `p`, print
1. `w`, where
1. `l`, 打印代码
1. `u`, up, 上一帧
1. `d`, down, 下一帧
1. `until`, 用于跳出循环
1. `until 5`,  运行到第5行
1. `return`, 运行到函数返回前
1. `retval`, 查看函数返回结果
1. `c`, continue, 继续执行
1. `b`, breakpoint, 列出所有断点
1. `b 5`, 在第五行添加断点
1. `q`, quit, 退出debug
1. `c`, clear, 删除所有断点
1. `c 1`, 删除编号为1的断点
---
---
># 终端
>1. `Alt Shift +/-` 打开垂直/水平窗格
---
---
# 账户密码
| 平台 | 账户  | 密码 |
| :----- | :---- | :---- |
| |益阳WIFI  | lyy123456 |
| gmail | lbxj2023@gmail.com |
| ikuuu | lbxj2023@gmail.com |
| ikuuu | 1183300812@qq.com |
| github | 1183300812@qq.com |
| overleaf | jx@nudt.edu.cn |



<font color=red>我是红色 `ctrl shift p`, markdown 打开侧面栏预览</font>  
<font color=#008000>我是绿色</font>  
<font color=yellow>我是黄色</font>  
<font color=Blue>我是蓝色</font>  
<font color= #871F78>我是紫色</font>  
<font color= #DCDCDC>我是浅灰色</font>  
<font size=5>我是尺寸</font>  
<u>带下划线文本</u>  
~~删除线~~  
*斜体*  
**加粗**

- 打印模型结构到文件
```python
from torchsummary import summary
model = MobileNetV2().to('cuda')
import sys
with open('summary.txt', 'w') as f:
    # 将stdout重定向到文件
    sys.stdout = f
    # 打印模型摘要
    summary(model, (3, 224, 224))   # print(model)
    # 将stdout恢复为标准输出
    sys.stdout = sys.__stdout__
```
- 打印模型参数
```python
model = MobileNetV2().to('cuda')
for name, shape in model.named_parameters():  
    print(f'{name}, {shape.size()}')
```
- 打印运行时间
```python
import time
if __name__ == '__main__':
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f'Run perf_counter: {end_time - start_time} s')
```

- 添加并打印python搜索路径
```python
import sys
# sys.path.append('/path/to/my_package')
print(sys.path)
```
- 替换transormer里的文件
```python
import importlib.util
# 1. 加载你本地的 modeling_llama.py 文件
spec = importlib.util.spec_from_file_location(
    "transformers.models.llama.modeling_llama", 
    "./mycode/modeling_llama.py"
)
custom_llama = importlib.util.module_from_spec(spec)
sys.modules["transformers.models.llama.modeling_llama"] = custom_llama
spec.loader.exec_module(custom_llama)
```
- 删除pip install的python包
```bash
python -m site
cd /usr/local/lib/python3.8/dist-packages
grep -r "/cephfs/juxin/smoothquant" . # 删除对应的文件或者行
```


### 表格
| 左对齐 | 右对齐 | 居中对齐 |
| :-----| ----: | :----: |
| 单元格 | 单元格 | 单元格 |
| 单元格 | 单元格 | 单元格 |



---
---
### 服务器记录
- ImageNet路径: `/data2/juxin`


# Activation Channel Scales

We provide the activation channel scales for OPT and BLOOM models at [Huggingface](https://huggingface.co/mit-han-lab/smoothquant-scales). We get those scales with 512 random sentences in the Pile validation set. You can use `../examples/smoothquant_opt_demo.ipynb` to test smoothing and quantizing those models.


from huggingface_hub import snapshot_download

# 下载整个仓库（包括 scales）
snapshot_download(repo_id="mit-han-lab/smoothquant-scales", local_dir="./smoothquant-scales")

torch.arange(35 * 29).reshape(35, 29)


Host 119.29.236.16
  HostName 119.29.236.16
  Port 6000
  User juxin
  IdentityFile "C:\Users\11833\.ssh\id_rsa"

Host A100
  HostName 10.210.22.155
  Port 31338
  User root
  IdentityFile "C:\Users\11833\.ssh\id_rsa"

Host A100-2
  HostName 10.210.22.142
  Port 32083
  User root
  IdentityFile "C:\Users\11833\.ssh\id_rsa"

Host eda
        HostName 10.220.7.250
        Port 10003
        User juxin