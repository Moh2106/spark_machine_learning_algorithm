package org.example;

import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class KMeansModelLearning {
    public static void main(String[] args) {
        SparkSession ss = SparkSession.builder().appName("K Means App").master("local[*]").getOrCreate();

        Dataset<Row> dataSet = ss.read().option("inferSchema", true).option("header", true).csv("Mall_Customers.csv");

        VectorAssembler assembler = new VectorAssembler().setInputCols(
                new String[]{"Age","Annual Income (k$)","Spending Score (1-100)"}
        ).setOutputCol("Features");

        Dataset<Row> assemblerDS = assembler.transform(dataSet);

        Dataset<Row>[] splits = assemblerDS.randomSplit(new double[]{0.8, 0.2}, 123);
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        KMeans kMeans = new KMeans().setK(3).setFeaturesCol("Features")
                .setPredictionCol("cluster");

        KMeansModel model = kMeans.fit(assemblerDS);
        Dataset<Row> transform = model.transform(assemblerDS);

    }
}
