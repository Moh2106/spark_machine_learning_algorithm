package org.example;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

// Press Shift twice to open the Search Everywhere dialog and type `show whitespaces`,
// then press Enter. You can now see whitespace characters in your code.
public class LinearRegressionModel {
    public static void main(String[] args) {
            SparkSession ss = SparkSession.builder().appName("Spark Machine Learning").master("local[*]").getOrCreate();

            Dataset<Row> dataSet = ss.read().option("inferSchema", true).option("header", true).csv("advertising.csv");

            VectorAssembler assembler = new VectorAssembler().setInputCols(
                    new String[]{"TV", "Radio", "Newspaper"}
            ).setOutputCol("Features");

            Dataset<Row> assemblerDS = assembler.transform(dataSet);

        Dataset<Row>[] splits = assemblerDS.randomSplit(new double[]{0.8, 0.2}, 123);
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        assemblerDS.printSchema();
        assemblerDS.show(10);

        LinearRegression model = new LinearRegression().setFeaturesCol("Features").setLabelCol("Sales");

        org.apache.spark.ml.regression.LinearRegressionModel lr = model.fit(trainingData);
        Dataset<Row> predictions = lr.transform(testData);
        predictions.show();


    }

}