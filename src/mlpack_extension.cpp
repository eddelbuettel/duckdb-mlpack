#define DUCKDB_EXTENSION_MAIN

#include "mlpack_extension.hpp"
#include "mlpack_utilities.hpp"

#include "mlpack_adaboost.hpp"
#include "mlpack_kmeans.hpp"
#include "mlpack_linear_regression.hpp"
#include "mlpack_logistic_regression.hpp"
#include "mlpack_random_forest.hpp"

#include <duckdb.hpp>
#include <duckdb/common/exception.hpp>
#include <duckdb/common/string_util.hpp>
#include <duckdb/function/scalar_function.hpp>
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include <duckdb/parser/parsed_data/create_table_function_info.hpp>

#include <mlpack.hpp> // mlpack

namespace duckdb {

// Function loading

static void LoadInternal(ExtensionLoader &loader) {
	auto &dbinstance = loader.GetDatabaseInstance();
	Connection con(dbinstance);

	// Register adaboost example train and prediction function
	{
		CreateTableFunctionInfo info(TableFunction("mlpack_adaboost_train",
												   { LogicalType::VARCHAR, LogicalType::VARCHAR,
													 LogicalType::VARCHAR, LogicalType::VARCHAR },
												   MlpackAdaboostTrainTableFunction,
												   MlpackTrainTableBindInt));
		FunctionDescription desc;
		desc.parameter_names = { "data", "labels", "parameters", "model" };
		desc.description     = "Trains adaboost classification of 'labels' given 'data' and 'parameters', and stores 'model'.";
		desc.examples        = { R"(SELECT * FROM mlpack_adaboost_train("data", "labels", "parameters", "model");)" };
		desc.categories      = { "mlpack", "machine learning", "classification" };
		info.descriptions.push_back(desc);
		loader.RegisterFunction(std::move(info));
	}
	{
		CreateTableFunctionInfo info(TableFunction("mlpack_adaboost_pred",
												   { LogicalType::VARCHAR, LogicalType::VARCHAR },
												   MlpackAdaboostPredictTableFunction,
												   MlpackPredictTableBindInt));
		FunctionDescription desc;
		desc.parameter_names = { "new_data", "model" };
		desc.description     = "Predicts classification given 'new_data' and previously-fit adaboost 'model'.";
		desc.examples        = { R"(SELECT * FROM mlpack_adaboost_pred("new_data", "model");)" };
		desc.categories      = { "mlpack", "machine learning", "classification" };
		info.descriptions.push_back(desc);
		loader.RegisterFunction(std::move(info));
	}

	// Register kmeans example train and prediction function
	{
		CreateTableFunctionInfo info(TableFunction("mlpack_kmeans",
												   { LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR },
												   MlpackKmeansTrainTableFunction,
												   MlpackUnsupervisedTrainTableBindInt));
		FunctionDescription desc;
		desc.parameter_names = { "data", "parameters", "model" };
		desc.description     = "Assigns clusters via k-means given 'data' and 'parameters', and stores 'model'.";
		desc.examples        = { R"(SELECT * FROM mlpack_kmeans("data", "parameters", "model");)" };
		desc.categories      = { "mlpack", "machine learning", "clustering" };
		info.descriptions.push_back(desc);
		loader.RegisterFunction(std::move(info));
	}

	// Register linear regression example fit and prediction
	{
		CreateTableFunctionInfo info(TableFunction("mlpack_linear_regression_fit",
												   { LogicalType::VARCHAR, LogicalType::VARCHAR,
													 LogicalType::VARCHAR, LogicalType::VARCHAR },
												   MlpackLinearRegressionTrainTableFunction,
												   MlpackTrainTableBindInt));
		FunctionDescription desc;
		desc.parameter_names = { "data", "responses", "parameters", "model" };
		desc.description     = "Fits linear regression of 'responses' given 'data' and 'parameters', and stores 'model'.";
		desc.examples        = { R"(SELECT * FROM mlpack_linear_regression_fit("data", "responses", "parameters", "model");)" };
		desc.categories      = { "mlpack", "machine learning", "regression" };
		info.descriptions.push_back(desc);
		loader.RegisterFunction(std::move(info));
	}
	{
		CreateTableFunctionInfo info(TableFunction("mlpack_linear_regression_pred",
												   { LogicalType::VARCHAR, LogicalType::VARCHAR },
												   MlpackLinearRegressionPredictTableFunction,
												   MlpackPredictTableBindInt));
		FunctionDescription desc;
		desc.parameter_names = { "new_data", "model" };
		desc.description     = "Predicts responses given 'new_data' and previously-fit linear regression 'model'.";
		desc.examples        = { R"(SELECT * FROM mlpack_linear_regression_pred("new_data", "model");)" };
		desc.categories      = { "mlpack", "machine learning", "regression" };
		info.descriptions.push_back(desc);
		loader.RegisterFunction(std::move(info));
	}

	// Register logistic regression example fit and prediction
	{
		CreateTableFunctionInfo info(TableFunction("mlpack_logistic_regression_fit",
												   { LogicalType::VARCHAR, LogicalType::VARCHAR,
													 LogicalType::VARCHAR, LogicalType::VARCHAR },
												   MlpackLogisticRegressionTrainTableFunction,
												   MlpackTrainTableBindInt));
		FunctionDescription desc;
		desc.parameter_names = { "data", "labels", "parameters", "model" };
		desc.description     = "Fits logistic regression classification of 'labels' given 'data' and 'parameters', and stores 'model'.";
		desc.examples        = { R"(SELECT * FROM mlpack_logistic_regression_fit("data", "labels", "parameters", "model");)" };
		desc.categories      = { "mlpack", "machine learning", "classification" };
		info.descriptions.push_back(desc);
		loader.RegisterFunction(std::move(info));
	}
	{
		CreateTableFunctionInfo info(TableFunction("mlpack_logistic_regression_pred",
												   { LogicalType::VARCHAR, LogicalType::VARCHAR },
												   MlpackLogisticRegressionPredictTableFunction,
												   MlpackPredictTableBindInt));
		FunctionDescription desc;
		desc.parameter_names = { "new_data", "model" };
		desc.description     = "Predicts classification given 'new_data' and previously-fit logistic regression 'model'.";
		desc.examples        = { R"(SELECT * FROM mlpack_logistic_regression_pred("new_data", "model");)" };
		desc.categories      = { "mlpack", "machine learning", "classification" };
		info.descriptions.push_back(desc);
		loader.RegisterFunction(std::move(info));
	}

	// Register random forest example train and prediction function
	{
		CreateTableFunctionInfo info(TableFunction("mlpack_random_forest_train",
												   { LogicalType::VARCHAR, LogicalType::VARCHAR,
													 LogicalType::VARCHAR, LogicalType::VARCHAR },
												   MlpackRandomForestTrainTableFunction,
												   MlpackTrainTableBindInt));
		FunctionDescription desc;
		desc.parameter_names = { "data", "labels", "parameters", "model" };
		desc.description     = "Trains random forest classification of 'labels' given 'data' and 'parameters', and store in 'model'.";
		desc.examples        = { R"(SELECT * FROM mlpack_random_forest_train("data", "labels", "parameters", "model");)" };
		desc.categories      = { "mlpack", "machine learning", "classification" };
		info.descriptions.push_back(desc);
		loader.RegisterFunction(std::move(info));
	}
	{
		CreateTableFunctionInfo info(TableFunction("mlpack_random_forest_pred",
												   { LogicalType::VARCHAR, LogicalType::VARCHAR },
												   MlpackRandomForestPredictTableFunction,
												   MlpackPredictTableBindInt));
		FunctionDescription desc;
		desc.parameter_names = { "new_data", "model" };
		desc.description     = "Predicts classification given 'new_data' and previously-fit random forest 'model'.";
		desc.examples        = { R"(SELECT * FROM mlpack_random_forest_pred("new_data", "model");)" };
		desc.categories      = { "mlpack", "machine learning", "classification" };
		info.descriptions.push_back(desc);
		loader.RegisterFunction(std::move(info));
	}

	// Register settings 'verbose' and 'silent'
	auto &config = DBConfig::GetConfig(dbinstance);
	config.AddExtensionOption("mlpack_verbose", "Toggle whether to operate in verbose mode, default is false",
	                          LogicalType::BOOLEAN, Value(false));
	config.AddExtensionOption("mlpack_silent", "Toggle whether to operate in silent mode, default is false",
	                          LogicalType::BOOLEAN, Value(false));

	// Version helpers
	{
		CreateScalarFunctionInfo info(ScalarFunction("mlpack_mlpack_version",
													 {},
													 LogicalType::VARCHAR,
													 MlpackMlpackVersion));
		FunctionDescription desc;
		desc.description     = "Returns version number of mlpack library used.";
		desc.examples        = { R"(SELECT * FROM mlpack_mlpack_version();)" };
		desc.categories      = { "mlpack", "machine learning", "setup" };
		info.descriptions.push_back(desc);
		loader.RegisterFunction(std::move(info));
	}
	{
		CreateScalarFunctionInfo info(ScalarFunction("mlpack_armadillo_version",
													 {},
													 LogicalType::VARCHAR,
													 MlpackArmadilloVersion));
		FunctionDescription desc;
		desc.description     = "Returns version number of armadillo library used.";
		desc.examples        = { R"(SELECT * FROM mlpack_armadillo_version();)" };
		desc.categories      = { "mlpack", "machine learning", "setup" };
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
