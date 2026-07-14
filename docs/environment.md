# 环境配置说明

## Windows 端（跑实验）

| 组件 | 说明 |
|:----|:----|
| **Python** | 3.12.10，虚拟环境 `D:\ir-assignment\.venv\` |
| **Pyserini** | 2.3.0 |
| **JDK** | 21.0.11 LTS，`C:\Program Files\Eclipse Adoptium\jdk-21.0.11.10-hotspot` |
| **PyTorch** | 2.13.0+cu130（支持 5070 Ti GPU） |
| **faiss** | 1.14.3 |

### 预建索引（在 D 盘）
索引实际在 `D:\pyserini_cache\indexes\`，通过符号链接映射到 `C:\Users\MR\.cache\pyserini\indexes\`

### 代码中设置 JAVA_HOME
```python
os.environ['JAVA_HOME'] = r'C:\Program Files\Eclipse Adoptium\jdk-21.0.11.10-hotspot'
os.environ['PATH'] = os.path.join(os.environ['JAVA_HOME'], r'bin\server') + ';' + os.environ.get('PATH', '')
```

## Mac 端（写报告）
项目根目录：`~/Desktop/IR_Assignment/`
