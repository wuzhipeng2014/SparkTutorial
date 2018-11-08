package scala

import org.apache.spark.sql.SparkSession

import scala.ctrmodel.{GBDTCtrModel, GBTLRCtrModel}
import scala.evaluation.Evaluator
import scala.util.LibsvmUtils

object LRModelTest extends App {
  override def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("LRModelTrain")
      .master("local[2]")
      .getOrCreate()
    val evaluator = new Evaluator

    val trainDataPath="hdfs://localhost:9000/user/zhipengwu/zhipeng.wu/*.gz"
    /** 读取训练集 **/
    val trainData = LibsvmUtils.txt2LibsvmWithSample(spark, trainDataPath)
    val Array(trainingSamples, validationSamples)=trainData.randomSplit(Array(0.7, 0.3))


//    println("GBDT Ctr Prediction Model:")
//    val gbtModel = new GBDTCtrModel()
//    gbtModel.train(trainingSamples)
//    evaluator.evaluate(gbtModel.transform(validationSamples))


    println("GBDT+LR Ctr Prediction Model:")
    val gbtlrModel = new GBTLRCtrModel()
    gbtlrModel.train(trainingSamples)
    evaluator.evaluate(gbtlrModel.transform(validationSamples))

//    trainingSamples.take(40).map(line => println(line))

  }

}
