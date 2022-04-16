import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler


spark = SparkSession.builder.master("local[*]").getOrCreate()
spark.conf.set("spark.sql.repl.eagerEval.enabled", True)# Property used to format output tables better

df = spark.read.csv("./data/synth_composite.csv", inferSchema=True, header=True)

print(df.printSchema())

df = df.select("type", "amount", "oldbalanceOrg", "newbalanceOrig", "isFraud")

print(df.show(5))


train, test = df.randomSplit([0.7, 0.3], seed=645)

print(train.count())
print(test.count())

print(train.dtypes)

catCols = [x for (x, dataType) in train.dtypes if dataType == "string"]
numCols = [x for (x, dataType) in train.dtypes if ((dataType == "double") & (x != "isFraud"))]


# OHE----------------------------------------------

# Just check columns that are going to be processed
print(train.agg(F.countDistinct("type")).show())
print(train.groupby("type").count().show())

string_indexer = [StringIndexer(inputCol=x,
                                outputCol=x + "_StringIndexer",
                                handleInvalid="skip") for x in catCols]

one_hot_encoder = [OneHotEncoder(inputCols=[f"{x}_StringIndexer" for x in catCols],
                                 outputCols=[f"{x}_OneHotEncoder" for x in catCols])]

# Vector assembling ---------------------------------
assemblerInput = [x for x in numCols]
assemblerInput += [f"{x}_OneHotEncoder" for x in catCols]

vector_assembler = VectorAssembler(inputCols=assemblerInput,
                                   outputCol="VectorAssembler_features")



# Create Pipeline -----------------------------------

stages = []
stages += string_indexer
stages += one_hot_encoder
stages += [vector_assembler]


pipeline = Pipeline().setStages(stages)
model = pipeline.fit(train)

train_data_pipe = model.transform(train)

test_data_pipe = model.transform(test)

# Logistic Regression
train_data = train_data_pipe.select(F.col("VectorAssembler_features").alias("features"),
                                    F.col("isFraud").alias("label"))

test_data = test_data_pipe.select(F.col("VectorAssembler_features").alias("features"),
                                  F.col("isFraud").alias("label"))


lr_spark = LogisticRegression().fit(train_data)

print(f"Training AUC Score: {lr_spark.summary.areaUnderROC}")
# lr_spark.summary.pr.show()
pred = lr_spark.evaluate(test_data)

print(f"Test AUC Score: {pred.areaUnderROC}")
print(f"Test Accuracy: {pred.accuracy}")












