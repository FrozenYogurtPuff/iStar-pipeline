# 导入
# 初始化 Logger

#   BERT 深度学习

# Mock 文档读入

# 文本预处理
#   TODO: 从文档中提取文本
# paraList: list[str] = getTextFromFile(filename)

#   TODO: 从文本中抽取需求段落
# reqList: list[str] = getTextFromFile(paraText)

#   为段落使用spaCy分句
#       注意：reqList中，每一item为一段文本，则spacyList中每一item的spacy.tokens.Doc可能包含多句话

# 分析文本
#   主体分析
#       主体抽取
#           BERT 深度学习


# for-loop 以list[]为核心的中间形式存储
#           TODO: 依存成分

#           TODO: 包含依存成分的模板匹配

#       指代消解
#           TODO: 查找代词

#           TODO: 检索前文中出现的主体

#           TODO: 对代词按照单词成分进行匹配

#       主体关系
#           TODO: 模板匹配

#           TODO: 依存成分


#   意图分析
#       意图抽取与分类
#           BERT 深度学习


#           TODO: 核心意图 - 对结果基于依存分析的启发式规则进行意图构建
#           抽出一个可复用的函数

#           辅助意图
#               TODO: 结合核心识别结果与依存分析分割子句
# while could_split:
# sub_sent = doSplit()
#               TODO: 对子句进行 BERT

# Call 核心意图函数复用

#           预想对这些进行处理时，已经是嵌套式的成句，可以一次对主句和子句做后续分析
#           质量意图
#               TODO: 启发式质量相关词表

#               TODO: 基于词向量的质量词匹配

#           条件意图
#               TODO: 基于模板匹配的方案对深度学习结果进行筛选

#       意图关系

#           Dependency
#               TODO: 通过启发式规则（依存分析与模板匹配）查找意图对应的施事受事

#       条件约束
#           Qualification
#               来自质量意图的约束
#                   TODO: 根据启发式规则匹配质量意图所修饰的元素

#                   TODO: 将对应的 Task/Resource 与 Quality 联系

#               来自条件意图的约束
#                   TODO: 根据启发式规则匹配条件意图所修饰的核心意图

#                   TODO: 找到核心意图对应的任务，与条件意图形成 Quality 联系

#           Refinement
#               TODO: 使用模板匹配检测任务与资源存在的精化情况

#               TODO: 将父任务与子任务（或父资源与子资源）使用 Refinement 联系

#           NeededBy
#               TODO:查找 Dependency 中受事包含 Resource 的意图

#               TODO: 可以将该意图与 Resource 结点使用 NeededBy 联系

#           Contribution
#               TODO: 根据启发式规则匹配以形容性词汇出现的质量意图

#               TODO: 将对应的 Quality 与 Task/Resource 联系
