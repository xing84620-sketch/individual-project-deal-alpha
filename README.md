# Deal Alpha

我的第一个 GitHub 仓库。

我在学数据分析与可视化，这是我的课程个人项目仓库。

## 项目介绍

**并购消息的公告效应：一项基于事件研究法的探索性与预测性数据分析**

本项目面向关注并购消息的学生投资者，研究并购公告前后收购方与被收购方的市场反应，并尝试用历史同类交易特征解释和估计公告后的超额收益。

作者：xing。

## 计划实现的功能

- 使用事件研究法计算异常收益（AR）和累计异常收益（CAR），展示公告前后的市场反应。
- 对比收购方与被收购方，以及不同支付方式、竞购情况和监管条件下的历史案例。
- 探索交易规模、溢价率等特征与公告效应的关系，用交叉验证评估预测效果。
- 使用 Streamlit 展示分析图表，并计划加入大语言模型辅助解读。

## 数据与方法

计划从公开行情渠道获取价格数据，从公司公告和 SEC 文件核对事件日期与交易特征；数据来源、事件窗口与缺失值处理方式将在实现时记录。历史案例与报告中的交易事实仍需逐项核验。

开题报告附录中的问卷数据为模拟数据，仅用于演示分析流程，不能作为真实用户调研证据。

## 当前状态与运行方式

当前已加入课程部署实验的复利计算器，支持调整投资年限和年化收益率。并购分析功能仍在规划中。

在 Mac 终端进入项目目录后运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit Community Cloud 部署使用仓库 `xing84620-sketch/individual-project-deal-alpha`、分支 `main`，入口文件为 `app.py`。本实验无需密钥或外部数据。

## 仓库文件

- `README.md`：项目介绍、进度和使用说明。
- `.gitignore`：排除密钥、虚拟环境及缓存。
- `app.py`：交互式复利计算器。
- `requirements.txt`：应用运行依赖。

后续按数据获取、事件研究、预测评估、可视化应用的顺序逐步补充代码。本项目用于课程研究，输出不作为买卖建议。

## 云端部署

应用地址：https://xing84620-deal-alpha.streamlit.app

部署仓库为 `xing84620-sketch/individual-project-deal-alpha`，分支为 `main`，入口为 `app.py`。依赖已从项目虚拟环境导出并锁定版本。

更新应用后运行 `git add app.py requirements.txt README.md`、`git commit -m "更新应用"` 和 `git push origin main`，Streamlit Cloud 会自动更新。

课程验收仍需本人使用手机流量访问应用，并将应用网址发送到课程群。
