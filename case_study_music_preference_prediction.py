from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.clustering import KMeans

def read_data(spark):
    activity_file_path = 's3n://lululemon-case-study/activity/sample100mb.csv'
    metadata_file_path = 's3n://lululemon-case-study/newmetadata'

    activity_schema = StructType([
        StructField("user_id", StringType(), True),
        StructField("timestamp", IntegerType(), True),
        StructField("song_id", StringType(), True),
        StructField("date", IntegerType(), True)])

    metadata_schema = StructType([
        StructField("song_id", StringType(), True),
        StructField("artist_id", IntegerType(), True)])

    activity_df = spark.read.csv(activity_file_path,header=False,schema=activity_schema)
    metadata_df = spark.read.csv(metadata_file_path, header=False, schema=metadata_schema)

    return activity_df,metadata_df

def prepare_data(activity_df_in):
    # create temp view and generate frequency of user & song id combination
    activity_df_in.createTempView("activity_v")
    user_song_df = spark.sql("select user_id,song_id,count(1) as frequency from activity_v group by user_id,song_id")

    # add index integer columns to convert string to int -> later to be used for ALS model
    indexers = [StringIndexer(inputCol=column, outputCol=column + "_index").fit(user_song_df) for column in
                list(set(user_song_df.columns))]
    pipeline = Pipeline(stages=indexers)
    user_song_df_idx = pipeline.fit(user_song_df).transform(user_song_df)

    als = ALS(rank=10, maxIter=5, regParam=0.01, userCol="user_id_index", itemCol="song_id_index", ratingCol="frequency")
    als_model_prediction = als.fit(user_song_df_idx).transform(user_song_df_idx)

    print("ALS model prediction")
    als_model_prediction.show()

    assemble = VectorAssembler(inputCols=['prediction'], outputCol='features')
    user_song_data_als = assemble.transform(als_model_prediction)
    print("ALS transformed feature")
    user_song_data_als.show()

    return user_song_data_als,user_song_df_idx

def predict_data(df_in):
    # Trains a k-means model.
    kmeans = KMeans(featuresCol='features',predictionCol='prediction_cluster').setK(240)

    # Make predictions
    df_out = kmeans.fit(df_in).transform(df_in)
    df_out.show()
    return df_out

def final_solution(user_song_pred,user_song_df,metadata_df):
    user_song_pred.createTempView("user_song_pred_v")
    user_song_df.createTempView("user_song_df_v")
    metadata_df.createTempView("metadata_df_v")
    final_df = spark.sql("select uspv.song_id,uspv.user_id,mv.artist_id,uspv.prediction_cluster as pred_cluster \
                             from user_song_pred_v uspv \
                             left join metadata_df_v mv on uspv.song_id = mv.song_id")

    final_df.createTempView("final_df_v")
    count_unique_user_per_cluster = spark.sql("select pred_cluster,count(distinct(user_id)) as cnt_user from final_df_v group by pred_cluster ")
    print("count of unique user per cluster")
    count_unique_user_per_cluster.show()
    top_artist_per_cluster = spark.sql(
        "select pred_cluster,artist_id from (\
        select pred_cluster,artist_id,dense_rank() over(partition by pred_cluster order by artist_cnt desc) as top_artist \
        from (select pred_cluster,artist_id,count(1) as artist_cnt \
                from final_df_v \
                where artist_id is not null group by pred_cluster,artist_id)temp ) temp1 \
        where top_artist=1 order by pred_cluster")
    print("Top artist per cluster")
    top_artist_per_cluster.show()




if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Music Preference Prediction for user with spark") \
        .getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    activity_df_out,metadata_df_out = read_data(spark)
    user_song_als_out,user_song_idx_data = prepare_data(activity_df_out)
    cluster_prediction = predict_data(user_song_als_out)
    final_solution(cluster_prediction,user_song_idx_data,metadata_df_out)