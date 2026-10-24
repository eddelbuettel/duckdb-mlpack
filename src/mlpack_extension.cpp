#define DUCKDB_EXTENSION_MAIN

#include "mlpack_extension.hpp"
#include "mlpack_utilities.hpp"

#include "mlpack_adaboost.hpp"
#include "mlpack_bayesian_linear_regression.hpp"
#include "mlpack_decision_tree.hpp"
#include "mlpack_kmeans.hpp"
#include "mlpack_lars.hpp"
#include "mlpack_linear_regression.hpp"
#include "mlpack_linear_svm.hpp"
#include "mlpack_logistic_regression.hpp"
#include "mlpack_naive_bayes.hpp"
#include "mlpack_random_forest.hpp"
#include "mlpack_softmax_regression.hpp"

#include <duckdb.hpp>
#include <duckdb/common/exception.hpp>
#include <duckdb/common/string_util.hpp>
#include <duckdb/function/scalar_function.hpp>
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include <duckdb/parser/parsed_data/create_table_function_info.hpp>

#include <mlpack.hpp> // mlpack

namespace duckdb {

// Helper to build the SELECT example string used in function documentation
static string MlpackGenerateExample(const string &name, const vector<string> &parameter_names) {
	string example = "SELECT * FROM " + name + "(";
	for (idx_t i = 0; i < parameter_names.size(); i++) {
		if (i > 0) {
			example += ", ";
		}
		example += "\"" + parameter_names[i] + "\"";
	}
	return example + ");";
}

// Register a table function along with its documentation
static void RegisterMlpackTableFunction(ExtensionLoader &loader, string name, vector<LogicalType> arg_types,
                                        table_function_t function, table_function_bind_t bind,
                                        vector<string> parameter_names, string description, vector<string> categories) {
	FunctionDescription desc;
	desc.parameter_names = std::move(parameter_names);
	desc.description = std::move(description);
	desc.examples = {MlpackGenerateExample(name, desc.parameter_names)};
	desc.categories = std::move(categories);
	CreateTableFunctionInfo info(TableFunction(std::move(name), std::move(arg_types), function, bind));
	info.descriptions.push_back(std::move(desc));
	loader.RegisterFunction(std::move(info));
}

// Four-parameter supervised train/fit functions: data, labels|responses, parameters, model
static void RegisterTrainFunction(ExtensionLoader &loader, string suffix, table_function_t function,
                                  table_function_bind_t bind, const string &target, const string &description,
                                  const string &category) {
	RegisterMlpackTableFunction(
	    loader, "mlpack_" + suffix,
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR}, function, bind,
	    {"data", target, "parameters", "model"}, description, {"mlpack", "machine learning", category});
}

// Two-parameter prediction functions: new_data, model
static void RegisterPredictFunction(ExtensionLoader &loader, string suffix, table_function_t function,
                                    table_function_bind_t bind, const string &description) {
	RegisterMlpackTableFunction(loader, "mlpack_" + suffix, {LogicalType::VARCHAR, LogicalType::VARCHAR}, function,
	                            bind, {"new_data", "model"}, description,
	                            {"mlpack", "machine learning", "classification"});
}

// Function loading

static void LoadInternal(ExtensionLoader &loader) {
	auto &dbinstance = loader.GetDatabaseInstance();
	Connection con(dbinstance);

	// Register adaboost example train and prediction function
	RegisterTrainFunction(
	    loader, "adaboost_train", MlpackAdaboostTrainTableFunction, MlpackTrainTableBindInt, "labels",
	    "Trains adaboost classification of 'labels' given 'data' and 'parameters', and stores 'model'.",
	    "classification");
	RegisterPredictFunction(loader, "adaboost_pred", MlpackAdaboostPredictTableFunction, MlpackPredictTableBindInt,
	                        "Predicts classification given 'new_data' and previously-fit adaboost 'model'.");

	// Register kmeans example clustering function
	RegisterMlpackTableFunction(
	    loader, "mlpack_kmeans", {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
	    MlpackKmeansTrainTableFunction, MlpackUnsupervisedTrainTableBindInt, {"data", "parameters", "model"},
	    "Assigns clusters via k-means given 'data' and 'parameters', and stores 'model'.",
	    {"mlpack", "machine learning", "clustering"});

	// Register linear regression example fit and prediction
	RegisterTrainFunction(
	    loader, "linear_regression_fit", MlpackLinearRegressionTrainTableFunction, MlpackTrainTableBindInt, "responses",
	    "Fits linear regression of 'responses' given 'data' and 'parameters', and stores 'model'.", "regression");
	RegisterPredictFunction(loader, "linear_regression_pred", MlpackLinearRegressionPredictTableFunction,
	                        MlpackPredictTableBindInt,
	                        "Predicts responses given 'new_data' and previously-fit linear regression 'model'.");

	// Register logistic regression example fit and prediction
	RegisterTrainFunction(
	    loader, "logistic_regression_fit", MlpackLogisticRegressionTrainTableFunction, MlpackTrainTableBindInt,
	    "labels",
	    "Fits logistic regression classification of 'labels' given 'data' and 'parameters', and stores 'model'.",
	    "classification");
	RegisterPredictFunction(loader, "logistic_regression_pred", MlpackLogisticRegressionPredictTableFunction,
	                        MlpackPredictTableBindInt,
	                        "Predicts classification given 'new_data' and previously-fit logistic regression 'model'.");

	// Register random forest example train and prediction function
	RegisterTrainFunction(
	    loader, "random_forest_train", MlpackRandomForestTrainTableFunction, MlpackTrainTableBindInt, "labels",
	    "Trains random forest classification of 'labels' given 'data' and 'parameters', and store in 'model'.",
	    "classification");
	RegisterPredictFunction(loader, "random_forest_pred", MlpackRandomForestPredictTableFunction,
	                        MlpackPredictTableBindInt,
	                        "Predicts classification given 'new_data' and previously-fit random forest 'model'.");

	// Register bayesian linear regression example fit and prediction
	RegisterTrainFunction(
	    loader, "bayesian_linear_regression_fit", MlpackBayesianLinearRegressionTrainTableFunction,
	    MlpackTrainTableBindInt, "responses",
	    "Fits Bayesian linear regression of 'responses' given 'data' and 'parameters', and stores 'model'.",
	    "regression");
	RegisterPredictFunction(
	    loader, "bayesian_linear_regression_pred", MlpackBayesianLinearRegressionPredictTableFunction,
	    MlpackPredictTableBindInt,
	    "Predicts responses given 'new_data' and previously-fit Bayesian linear regression 'model'.");

	// Register lars example fit and prediction
	RegisterTrainFunction(loader, "lars_fit", MlpackLARSTrainTableFunction, MlpackTrainTableBindInt, "responses",
	                      "Fits LARS regression of 'responses' given 'data' and 'parameters', and stores 'model'.",
	                      "regression");
	RegisterPredictFunction(loader, "lars_pred", MlpackLARSPredictTableFunction, MlpackPredictTableBindInt,
	                        "Predicts responses given 'new_data' and previously-fit LARS 'model'.");

	// Register naive bayes example train and prediction function
	RegisterTrainFunction(
	    loader, "naive_bayes_train", MlpackNaiveBayesTrainTableFunction, MlpackTrainTableBindInt, "labels",
	    "Trains naive bayes classification of 'labels' given 'data' and 'parameters', and stores 'model'.",
	    "classification");
	RegisterPredictFunction(loader, "naive_bayes_pred", MlpackNaiveBayesPredictTableFunction, MlpackPredictTableBindInt,
	                        "Predicts classification given 'new_data' and previously-fit naive bayes 'model'.");

	// Register decision tree example train and prediction function
	RegisterTrainFunction(
	    loader, "decision_tree_train", MlpackDecisionTreeTrainTableFunction, MlpackTrainTableBindInt, "labels",
	    "Trains decision tree classification of 'labels' given 'data' and 'parameters', and stores 'model'.",
	    "classification");
	RegisterPredictFunction(loader, "decision_tree_pred", MlpackDecisionTreePredictTableFunction,
	                        MlpackPredictTableBindInt,
	                        "Predicts classification given 'new_data' and previously-fit decision tree 'model'.");

	// Register softmax regression example fit and prediction function
	RegisterTrainFunction(
	    loader, "softmax_regression_fit", MlpackSoftmaxRegressionTrainTableFunction, MlpackTrainTableBindInt, "labels",
	    "Fits softmax regression classification of 'labels' given 'data' and 'parameters', and stores 'model'.",
	    "classification");
	RegisterPredictFunction(loader, "softmax_regression_pred", MlpackSoftmaxRegressionPredictTableFunction,
	                        MlpackPredictTableBindInt,
	                        "Predicts classification given 'new_data' and previously-fit softmax regression 'model'.");

	// Register linear svm example train and prediction function
	RegisterTrainFunction(
	    loader, "linear_svm_train", MlpackLinearSVMTrainTableFunction, MlpackTrainTableBindInt, "labels",
	    "Trains linear svm classification of 'labels' given 'data' and 'parameters', and stores 'model'.",
	    "classification");
	RegisterPredictFunction(loader, "linear_svm_pred", MlpackLinearSVMPredictTableFunction, MlpackPredictTableBindInt,
	                        "Predicts classification given 'new_data' and previously-fit linear svm 'model'.");

	// Register settings 'verbose' and 'silent'
	auto &config = DBConfig::GetConfig(dbinstance);
	config.AddExtensionOption("mlpack_verbose", "Toggle whether to operate in verbose mode, default is false",
	                          LogicalType::BOOLEAN, Value(false));
	config.AddExtensionOption("mlpack_silent", "Toggle whether to operate in silent mode, default is false",
	                          LogicalType::BOOLEAN, Value(false));

	// Version helpers
	{
		CreateScalarFunctionInfo info(
		    ScalarFunction("mlpack_mlpack_version", {}, LogicalType::VARCHAR, MlpackMlpackVersion));
		FunctionDescription desc;
		desc.description = "Returns version number of mlpack library used.";
		desc.examples = {R"(SELECT mlpack_mlpack_version();)"};
		desc.categories = {"mlpack", "machine learning", "setup"};
		info.descriptions.push_back(desc);
		loader.RegisterFunction(std::move(info));
	}
	{
		CreateScalarFunctionInfo info(
		    ScalarFunction("mlpack_armadillo_version", {}, LogicalType::VARCHAR, MlpackArmadilloVersion));
		FunctionDescription desc;
		desc.description = "Returns version number of armadillo library used.";
		desc.examples = {R"(SELECT mlpack_armadillo_version();)"};
		desc.categories = {"mlpack", "machine learning", "setup"};
		info.descriptions.push_back(desc);
		loader.RegisterFunction(std::move(info));
	}
}

void MlpackExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string MlpackExtension::Name() {
	return "mlpack";
}

std::string MlpackExtension::Version() const {
#ifdef EXT_VERSION_MLPACK
	return EXT_VERSION_MLPACK;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(mlpack, loader) {
	duckdb::LoadInternal(loader);
}

DUCKDB_EXTENSION_API const char *mlpack_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
