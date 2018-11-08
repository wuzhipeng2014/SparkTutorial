import org.apache.spark.SparkContext
import org.apache.spark.SparkConf


object test extends App {


  override def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("appname").setMaster("local")
    val sc = new SparkContext(conf)

    val data = Array(1, 2, 3, 4, 5)
    val distData = sc.parallelize(data)
    print(distData.first())
    // Load training data
    //    val training = sc.textFile("src/main/resources/citys.txt")
    val training = sc.textFile("hdfs://localhost:9000/user/zhipengwu/zhipeng.wu/*.gz")


//        val examples=training.sample(false, 1).collect
    val examples = training.takeSample(false, training.count().toInt)
    sc.makeRDD(examples)

    examples.map(line => println(line))


    print("test ")
    sc.stop()
  }
}