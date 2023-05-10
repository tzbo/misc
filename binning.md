  
- ### 一、前言
	- 为什么要分箱，参考：
		1. [机器学习为什么要进行变量分箱？分箱的应用场景有哪些？](https://zhuanlan.zhihu.com/p/376884531)
		2. [变量分箱方法](https://zhuanlan.zhihu.com/p/503235392)

	- 优化建议
	  对于写PySpark的同学必须牢记一点：尽量少写自定义的udf，特别是3之前的版本，因为会python和java之前会进行数据数据序列化和反序列化，数据量比较大的情况下特别影响性能；3之后可以使用pandas udf，对性能有比较大的提升，可以参考： [Introducing Pandas UDF for PySpark](https://www.databricks.com/blog/2017/10/30/introducing-vectorized-udfs-for-pyspark.html)   
	  
	- Spark优化总结
	  id:: 6440dd47-9dff-4e2a-8379-a1ee06cf9046
		- pyspark相关
		  id:: 6440ddf0-b36d-4f7a-b235-a4638c13701b
			- 尽量不要使用RDD相关的方法，除非必要，尽量使用DataFrame的方法
			- RDD方法会进行立即计算
			- RDD方法会进行python和java之间数据序列化和反序列化
			- 如果非要使用的话，尽量使用arrow特性（spark3开始全面支持，spark2是experimental）
			- [PySpark Usage Guide for Pandas with Apache Arrow](https://spark.apache.org/docs/3.0.1/sql-pyspark-pandas-with-arrow.html)
		- 分组计算
			- 如果需要进行groupBy进行计算，需要分析是否需要将整个group cache在内存中
			- 如果不需要cache，则基本没有问题（groupBy会使用ExternalXXX相关类）
			- 如果需要cache，则可以使用numpy的memmap将数据保存到磁盘，不然很容易出现OOM
			- 如果使用pandas_udf需要注意内存，spark会一次取出所有group数据
		- 对于列数行多的表
			- 优先考虑使用列转行，可以使用stack函数
			- 对于列转行，尽量不要使用map相关函数，参考： pyspark相关
			- 如果sql较多，可能会出现
		- 其他
			- 如果要进行统计，尽量使用sql统计，且尽量使用单sql
			- sql执行会进行优化，然后生成代码，如果列数过多可能会超过限制（64kb）
	    
- ### 二、数据预处理
	- 列转行
		1. 为什么要做列转行？对于分箱我们知道，就是将数据进行排序后按照某种规则进行分段。如果将整张别统一处理的话，我们只能串行的单列处理：排序Column1，再分箱；排序Column2，再分箱...，不能利用spark的多核计算能力。
		2. 如何进行行转列呢？前面我们已经提到过，尽量少用udf，我们可以使用spark sql中的stack函数
		  ```python
def unpivot_df(self, df: DataFrame, y_col):
cols_len = len(df.columns) - 1
cols_expr = []
for i in range(cols_len):
cols_expr.append('{}'.format(i))
cols_expr.append('cast({} as double)'.format(df.columns[i]))

unpivot_expr = 'stack({},{}) as (col,value)'.format(cols_len, ','.join(cols_expr))
unpivot_df = df.select(F.expr(unpivot_expr), y_col)
return unpivot_df
		  ```
		3. 将数据进行行转列后，数据表变成了3列：(col，value，label)

	- 排序
		1. 其实排序比较简单，既然已经行转列后，按col分组排序即可，但是排序之后怎么办？
		2. 写过分箱算法的同学都知道，单列数据的分箱很难使用多核（并行）计算，至少设计起来要麻烦很多；即使设计出了多核分箱算法，我们还需要spark的分布式计算能力
		3. 那排序之后的数据如何分箱呢？这里我们不考虑多核分箱算法，我们只考虑如何利用spark的分布式计算能力，排序后的数据不能简单的全collect到driver计算，我们需要将分箱计算分到各个executor计算
		4. 我们可以按列做partition，然后在partition里面做排序，然后在partition中做map操作
		  ```
unpivot_df.repartition('col').sortWithinPartitions('col', 'value')
		  ```
		5. 为什么要用col和value做排序？问题先放在这里

- ### 三、分箱
	- 算法设计分析
		1. 通过前面的数据处理，我们真正到了分箱环节，在每个partition中我们有了按col和value排序好的数据
		2. 这里有个问题，我们可以将partition中的数据全都读到内存中进行分箱计算吗？如果你的内存足够大，那肯定没有问题。但是现实中我们的内存是有限的，比如可能分到executor的内存只有几G，甚至更少。同时executor可以用于代码执行的内存是有一定比例的： [Apache Spark Memory Management](https://medium.com/analytics-vidhya/apache-spark-memory-management-49682ded3d42)
		3. 另外，对于单行数据，我们可以估算需要的内存，比如：数据有1亿行，每value按8字节计算，大概需要700多M，我们的数据有3列，大概就需要2个多G（注意是估算），再按照executor分给代码的内存比例，则executor单个core可能需要5个G左右。所以如果每个partition中的列如果多的话肯定会有内存问题
		4. 我们首先想到的是可以按列处理分箱，这就是我们问什么按col和value做排序
		5. 如果分配的数据一列都放不下呢？注意每列数据需要3列来表示（行转列），这里我们可以使用numpy的memmap list，将数据放到磁盘
		6. 另外这里有个优化，计算分箱（ks、卡方等）的时候最频繁使用的数据其实是标签统计，所以这里我们尽量将标签统计放到内存中，可以看到`column_data_iter`每次读取一列数据
		  ```python
def column_data_iter(partition_iter, memory, count, single_value):
filename = str(id(partition_iter))
data_file, data = new_memmap_list('binning_data', filename, np.float64, (count, 2))
single_file, single_data = new_memmap_list('binning_single', filename, np.float64, (count, 2))
acc_file, label_acc = new_list(memory*0.6, 'binning_acc', filename, np.int32, (count, 2))
column_index = -1
last_acc = [0, 0]
data_count = 0
single_count = 0
for row in partition_iter:
label = row[2]
column_index = row[0] if column_index == -1 else column_index
if row[0] == column_index:
if row[1] == single_value:
single_data[single_count] = row[1:]
single_count += 1
else:
data[data_count] = row[1:]
another = BAD if label == GOOD else GOOD
label_acc[data_count][label] = last_acc[label] + 1
label_acc[data_count][another] = last_acc[another]
last_acc = label_acc[data_count]
data_count += 1
else:
yield column_index, data_count, data, single_data, label_acc

column_index = row[0]
last_acc = [0, 0]
data_count = 0
single_count = 0
if row[1] == single_value:
single_data[single_count] = row[1:]
single_count += 1
else:
data[data_count] = row[1:]
another = BAD if label == GOOD else GOOD
label_acc[data_count][label] = last_acc[label] + 1
label_acc[data_count][another] = last_acc[another]
last_acc = label_acc[data_count]
data_count += 1

# last row
yield column_index, data_count, data, single_data, label_acc

del data
del single_data
del label_acc
os.remove(data_file)
os.remove(single_file)
if acc_file is not None:
os.remove(acc_file)
		  ```

	- 算法性能测试
	  我自己测试的时候集群也都是用的虚拟机，机器都比较烂，测试下来的性能如下：  
		1. 500列50w需要5分钟左右，并行task数量为24
		2. 5000列50w需要45-50分钟左右，并行task为26
		3. 500列1000w需要90分钟左右，并行task也是20多个

- ### 四、写在最后
	1. 对于类似分箱这样的算法，在spark里需要做group操作，因为要进行shuffle，PySpark本身已经做了比较多的优化，比如：外排序等，但是还是会经常出现memory问题
	2. 单列数据比较大的情况分箱也可以做多核计算，这个有精力的同学可以分享下
	3. [完整算法](https://github.com/tzbo/misc/blob/main/binning.py)

