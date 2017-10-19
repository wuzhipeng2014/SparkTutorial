package GetStarting.ML;

import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.List;

/**
 * Created by zhipengwu on 17-10-18.
 */
public class FeatureManipulation {

    public static void main(String[] args) {

        SparkSession spark = SparkSession.builder().master("local").appName("Simple Application").getOrCreate();

        List<Row> data = Arrays.asList(RowFactory.create(0.0, "Hi I heard about Spark"),
                RowFactory.create(0.0, "I wish Java could use case classes"),
                RowFactory.create(1.0, "Logistic regression models are neat"));
        StructType schema = new StructType(
                new StructField[] { new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
                        new StructField("sentence", DataTypes.StringType, false, Metadata.empty()) });

        Dataset<Row> sentenceData = spark.createDataFrame(data, schema);

        Tokenizer tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words");
        Dataset<Row> wordsData = tokenizer.transform(sentenceData);

        int numFeatures = 20;
        HashingTF hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures")
                .setNumFeatures(numFeatures);

        Dataset<Row> featurizedData = hashingTF.transform(wordsData);
        // alternatively, CountVectorizer can also be used to get term frequency vectors
        System.out.println("HashingTF转换后的向量");
        featurizedData.show();
        Column rawFeatures = featurizedData.col("rawFeatures");

        System.out.println(rawFeatures.getItem(1).as(Encoders.STRING()));

        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
        IDFModel idfModel = idf.fit(featurizedData);

        Dataset<Row> rescaledData = idfModel.transform(featurizedData);
        Dataset<Row> selectResult = rescaledData.select("label", "features");
        System.out.println("IDF 转换后的向量");
        selectResult.show();
//        selectResult.write().save("/home/zhipengwu/secureCRT/selectResult.txt");

    }
}
