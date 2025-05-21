# mindspore_op_plugin

#### 介绍
op_plugin for mindspore

#### 软件架构
软件架构说明


#### 安装教程
安装前请确保环境中存在pytorch(cpu版本)：
1.  将 `patch/op_plugin.patch` 打在最新 MindSpore 源码上，重新编译安装MindSpore;
2.  `bash build.sh` 构建 op_plugin；
3.  `source env.source` 设置环境变量。

#### 使用说明

1.  测试CPU算子 `test/test_acos.py`.

#### 注意事项

1.  若需要和 pytorch 共进程运行，则只能安装 torch 2.1.0 版本，否则会导致共享库冲突，产生找不到符号表等未定义的行为。

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
