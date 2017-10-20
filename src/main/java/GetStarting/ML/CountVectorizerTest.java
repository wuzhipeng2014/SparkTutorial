package GetStarting.ML;

import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.List;

/**
 * Created by zhipengwu on 17-10-20.
 * 稀疏表示:　用尽可能少的字符表示文档
 * +---------------+-------------------------+
 |text           |feature                  |
 +---------------+-------------------------+
 |[a, b, c]      |(4,[0,1,2],[1.0,1.0,1.0])|
 |[a, b, b, c, a]|(4,[0,1,2],[2.0,2.0,1.0])|
 +---------------+-------------------------+
 feature列
 4. 表示词汇表中有四个字符
 [0,1,2] 表示文档中出现了三个字符,按出现频次分别用0,1,2替换
 [2.0,2.0,1.0] 表示三个字符分别出现了多少次
 */
public class CountVectorizerTest {
    public static void main(String[] args) {

        SparkSession spark = SparkSession.builder().master("local").appName("Simple Application").getOrCreate();

        // Input data: Each row is a bag of words from a sentence or document.
        List<Row> data = Arrays.asList(RowFactory.create(Arrays.asList("a", "b", "c","d")),
                RowFactory.create(Arrays.asList("a", "b", "b", "c", "a","a","d")));
        StructType schema = new StructType(new StructField[] {
                new StructField("text", new ArrayType(DataTypes.StringType, true), false, Metadata.empty()) });
        Dataset<Row> df = spark.createDataFrame(data, schema);

        // fit a CountVectorizerModel from the corpus
        CountVectorizerModel cvModel = new CountVectorizer().setInputCol("text").setOutputCol("feature").setVocabSize(4)
                .setMinDF(2).fit(df);

        // alternatively, define CountVectorizerModel with a-priori vocabulary
        CountVectorizerModel cvm = new CountVectorizerModel(new String[] { "a", "b", "c","d" }).setInputCol("text")
                .setOutputCol("feature");

        cvModel.transform(df).show(false);

        cvm.transform(df).show(false);
    }
}
