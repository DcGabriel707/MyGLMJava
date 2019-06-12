import com.github.chen0040.data.evaluators.BinaryClassifierEvaluator;
import com.github.chen0040.data.evaluators.ClassifierEvaluator;
import com.github.chen0040.data.frame.DataColumn;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataQuery;
import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.data.utils.TupleTwo;
import com.github.chen0040.glm.enums.GlmSolverType;
import com.github.chen0040.glm.solvers.Glm;
import com.github.chen0040.glm.solvers.OneVsOneGlmClassifier;

import java.io.*;

public class GLMTest {
    public static void main(String[] args) throws IOException {
        //***************************Create Dataframe***************************//
        int col_admit = 0;
        int col_gre = 1;
        int col_gpa = 2;
        int col_rank2 = 3;
        int col_rank3 = 4;
        int col_rank4 = 5;

        boolean skipFirstLine = true;
        String columnSplitter = ",";
        InputStream inputStream = new FileInputStream("binaryTestCategorical.csv");
        DataFrame frame = DataQuery.csv(columnSplitter, skipFirstLine)
                .from(inputStream)
                //.selectColumn(col_rank).asCategory().asInput("rank")
                //.selectColumn(col_rank).asNumeric().asInput("rank")
                //.selectColumn(col_gpa).transform(cell -> cell.equals("1") ? 1.0 : 0.0).asInput("gpa")

                .selectColumn(col_rank2).transform(cell -> cell.equals("1") ? 1.0 : 0.0).asInput("rank2")
                .selectColumn(col_rank3).transform(cell -> cell.equals("1") ? 1.0 : 0.0).asInput("rank3")
                .selectColumn(col_rank4).transform(cell -> cell.equals("1") ? 1.0 : 0.0).asInput("rank4")

                .selectColumn(col_gpa).asNumeric().asInput("gpa")
                .selectColumn(col_gre).asNumeric().asInput("gre")

                .selectColumn(col_admit).transform(cell -> cell.equals("1") ? 1.0 : 0.0).asOutput("admit")


                .build();


        frame.stream().forEach(r -> System.out.println("row: " + r));
        System.out.println("categorical column count: " + frame.getAllColumns().stream().filter(DataColumn::isCategorical).count());
        System.out.println("numerical column count: " + frame.getAllColumns().stream().filter(DataColumn::isNumerical).count());
        System.out.println("00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");
        for (DataRow r : frame) {
            System.out.println("row: " + r);
        }

        //***************************Perform GLM***************************//

        for (int i = 0; i < frame.rowCount(); ++i) {
            DataRow row = frame.row(i);
            String targetColumn = row.getTargetColumnNames().get(0);
            //row.setTargetCell(targetColumn, row.getTargetCell(targetColumn) == -1 ? 0 : 1); // change output from (-1, +1) to (0, 1)
        }


        //splits dataframe into two parts. one will be used to train the program. the other is for testing the algorithm
        //TupleTwo<DataFrame, DataFrame> miniFrames = frame.shuffle().split(0.9); //shuffles the observations. gives different coefficient everytime
        //TupleTwo<DataFrame, DataFrame> miniFrames = frame.split(.99);
        //DataFrame trainingData = miniFrames._1();
        //DataFrame crossValidationData = miniFrames._2();


        DataFrame trainingData = frame;
        DataFrame crossValidationData = frame;

        Glm algorithm = Glm.logistic();
        algorithm.setSolverType(GlmSolverType.GlmIrls);
        algorithm.fit(trainingData);

        double threshold = 1.0;
        for (int i = 0; i < trainingData.rowCount(); ++i) {
            double prob = algorithm.transform(trainingData.row(i));
            if (trainingData.row(i).target() == 1 && prob < threshold) {
                threshold = prob;
            }
        }
        //logger.info("threshold: {}",threshold);


        BinaryClassifierEvaluator evaluator = new BinaryClassifierEvaluator();

        for (int i = 0; i < crossValidationData.rowCount(); ++i) {
            double prob = algorithm.transform(crossValidationData.row(i));
            boolean predicted = prob > 0.5;
            boolean actual = crossValidationData.row(i).target() > 0.5;
            evaluator.evaluate(actual, predicted);
            System.out.println("probability of positive: " + prob);
            System.out.println("predicted: " + predicted + "\tactual: " + actual);
        }

        evaluator.report();
        System.out.println("Coefficients: " + algorithm.getCoefficients());
    }



}


