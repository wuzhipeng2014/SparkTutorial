package GetStarting;


import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.SparkSession;

/**
 * Created by zhipengwu on 17-10-17.
 */
public class HelloSpark {

    public static void main(String[] args) {





        String logFile = "src/main/resources/citys.txt"; // Should be some file on your system
        SparkSession spark = SparkSession.builder().master("local").appName("Simple Application").getOrCreate();
        Dataset<String> logData = spark.read().textFile(logFile).cache();



        spark.stop();
    }

}
