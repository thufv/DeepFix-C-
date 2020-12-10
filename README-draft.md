# DeepFix-C#

本项目对 DeepFix 进行了修改，使之能处理 C# 代码（仅 typo）。主要修改如下：

* 添加 C# 实验数据集；
* 添加 C# tokenizer；
* 针对我们的训练数据集，重新定义划分训练集和验证集的方式；
* 针对我们的测试数据集，重写测试脚本；
* 重写 end-to-end 脚本，其中将原来的 `source activate deepfix` 替换为 `conda activate deepfix`，将原来的 `pip install` 替换为 `conda install`。

## 环境配置

```
$ cat /usr/local/cuda/version.txt
CUDA Version 10.0.130
$ conda --version
conda 4.9.1
$ mcs --version
Mono C# compiler version 4.6.2.0
```

## 复现步骤

`bash e2e-cs.sh` 即可。此即 end-to-end 脚本，负责创建 conda 虚拟环境并安装所需的包，然后依次生成训练数据、训练（one-fold）、测试。测试结果输出到 `proc_cs.json`。

## 结果

* Ubuntu 18.04.3 LTS
* Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz
* MemTotal: 65840148 kB (~64 GB)
* GPU 0~6: GeForce RTX 2080 Ti

### 小

(TrainSize, ValidationSize) == (3870, 1286)

len(CompletelyFixed) == 15 (all are of CS0022)

See `res/small/`. Only those are completely fixed are included.

Fix example: In `res/small/CS0022/0/`,

* `0.cs` is before fix;
* `1.cs` is after fix;
* `a.json` contains more info;

```json
{
  "raw_code": "using System;\nusing System.Linq;\npublic class Test {\n  static int[] a = new int[10];\n  public static void Main() {\n    for (int i = 0; i != 10; i++) {\n      a[i, 0] = i + 99\n      /* updated */\n      ;\n    }\n  }\n}\n",
  "raw_error_count": 1,
  "final_code": "\nusing System ; \nusing System . Linq ; \npublic class Test { \nstatic int [,] a = new int [ 0 , 0 ]; \npublic static void Main (){ \nfor ( int i = 0 ; i != 0 ; i ++){ \na [ i , 0 ]= i + 0 \n;}}} ",
  "final_error_count": 0,
  "iteration_count": 1,  // how many fix iterations needed from raw_code to final_code
}
```

Training process:

See `res/small/train.xlsx`. Each row represents an epoch.

### 大

(TrainSize, ValidationSize) == (14539, 5056)

len(CompletelyFixed) == 4 (all are of CS0022)

See `res/big/`.
