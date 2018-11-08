package scala.util

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.functions.rand
/**
  * Created by tianle.li on 2017/9/28.
  */
object LibsvmUtils {

  def parseLibSVM(line: (String, String)): (String, String, Double, Array[Int], Array[Double]) = {
    val itemsAll = line._2.split(' ')
    val label = if (itemsAll.head.toDouble == 0) {
      0
    } else {
      1
    }
    val items = itemsAll.tail.sortWith { case (a, b) => a.split(":")(0).toInt < b.split(":")(0).toInt }

    val (indices, values) = items.filter(_.nonEmpty).map { item =>
      val indexAndValue = item.split(':')
      val index = indexAndValue(0).toInt //- 1 // Convert 1-based indices to 0-based.
    val value = indexAndValue(1).toDouble
      (index, value)
    }.unzip

    // check if indices are one-based and in ascending order
    var previous = -1
    var i = 0
    val indicesLength = indices.length
    while (i < indicesLength) {
      val current = indices(i)
      require(current > previous, s"indices should be one-based and in ascending order;"
        + " found current=$current, previous=$previous; line=\"$line\"")
      previous = current
      i += 1
    }

    (line._1, line._2, label, indices.toArray, values.toArray)
  }

  def computeNumFeatures(rdd: RDD[(String, String, Double, Array[Int], Array[Double])]): Int = {
    rdd.map { case (keyid, libSvm, label, indices, values) =>
      indices.lastOption.getOrElse(0)
    }.reduce(math.max) + 1
  }

  def get_df_data(rdd: RDD[(String, String)], numFeatures: Int): RDD[(String, String, LabeledPoint)] = {
    val parsed = rdd.map(parseLibSVM)
    // Determine number of features.
    val d = if (numFeatures > 0) {
      numFeatures
    } else {
      parsed.persist(StorageLevel.MEMORY_ONLY)
      computeNumFeatures(parsed)
    }
    val res = parsed.map { case (keyid, libSvm, label, indices, values) =>
      (keyid, libSvm, LabeledPoint(label, Vectors.sparse(d, indices, values)))
    }
    res
  }

  def txt2Libsvm(spark: SparkSession, path: String): DataFrame = {
    val txtRdd = spark.read.textFile(path).rdd
    val rddItem = txtRdd.map(line => line.trim.split('\t')).filter(line => !(line.isEmpty || line.startsWith("#"))).map(ary => (ary(0), ary(1)))
    val res = get_df_data(rddItem, 0)

    import spark.implicits._
    res.map { case (kv, libSvm, v) => (kv, libSvm, v.label, v.features) }.toDF("keyid", "libsvm", "label", "features")
  }
  def txt2LibsvmWithSample(spark: SparkSession, path: String): DataFrame = {
//    val txtRdd = spark.read.textFile(path).sample(false,1).rdd
    val txtRdd = spark.read.textFile(path).sample(false,1).orderBy(rand()).rdd
    val rddItem = txtRdd.map(line => line.trim.split('\t')).filter(line => !(line.isEmpty || line.startsWith("#"))).map(ary => (ary(0), ary(1)))
    val res = get_df_data(rddItem, 0)

    import spark.implicits._
    res.map { case (kv, libSvm, v) => (kv, libSvm, v.label, v.features) }.toDF("keyid", "libsvm", "label", "features")
  }
}