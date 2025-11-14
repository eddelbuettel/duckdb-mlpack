
#include <mlpack/core/util/version.hpp>
#include <armadillo_bits/arma_version.hpp>

#include "mlpack_model_data.hpp"
#include "mlpack_utilities.hpp"

namespace duckdb {

unique_ptr<FunctionData> MlpackTrainTableBindInt(ClientContext &context, TableFunctionBindInput &input,
                                                 vector<LogicalType> &return_types, vector<string> &names) {
	auto resdata = make_uniq<MlpackModelData>();
	resdata->features = input.inputs[0].GetValue<std::string>();
	resdata->labels = input.inputs[1].GetValue<std::string>();
	resdata->parameters = input.inputs[2].GetValue<std::string>();
	resdata->model = input.inputs[3].GetValue<std::string>();
	names = {"predicted"};
	return_types = {LogicalType::INTEGER};
	resdata->return_types = return_types;
	resdata->names = names;
	return std::move(resdata);
}

unique_ptr<FunctionData> MlpackPredictTableBindInt(ClientContext &context, TableFunctionBindInput &input,
                                                   vector<LogicalType> &return_types, vector<string> &names) {
	auto resdata = make_uniq<MlpackModelData>();
	resdata->features = input.inputs[0].GetValue<std::string>();
	resdata->model = input.inputs[1].GetValue<std::string>();
	names = {"predicted"};
	return_types = {LogicalType::INTEGER};
	resdata->return_types = return_types;
	resdata->names = names;
	return std::move(resdata);
}

unique_ptr<FunctionData> MlpackTrainTableBindDouble(ClientContext &context, TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types, vector<string> &names) {
	auto resdata = make_uniq<MlpackModelData>();
	resdata->features = input.inputs[0].GetValue<std::string>();
	resdata->labels = input.inputs[1].GetValue<std::string>();
	resdata->parameters = input.inputs[2].GetValue<std::string>();
	resdata->model = input.inputs[3].GetValue<std::string>();
	names = {"predicted"};
	return_types = {LogicalType::DOUBLE};
	resdata->return_types = return_types;
	resdata->names = names;
	return std::move(resdata);
}

unique_ptr<FunctionData> MlpackPredictTableBindDouble(ClientContext &context, TableFunctionBindInput &input,
                                                      vector<LogicalType> &return_types, vector<string> &names) {
	auto resdata = make_uniq<MlpackModelData>();
	resdata->features = input.inputs[0].GetValue<std::string>();
	resdata->model = input.inputs[1].GetValue<std::string>();
	names = {"predicted"};
	return_types = {LogicalType::DOUBLE};
	resdata->return_types = return_types;
	resdata->names = names;
	return std::move(resdata);
}

void MlpackMlpackVersion(DataChunk &args, ExpressionState &state, Vector &result) {
	// there are no argument-less UDFs, or at least I found no easy way to register one
	// so we fake a one-element argument vector and the Excecute() make one call where
	// our constant payload is returned
	Vector ignored_vector(LogicalType::VARCHAR, false, false, 1);
	UnaryExecutor::Execute<string_t, string_t>(ignored_vector, result, 1, [&](string_t unused) {
		return StringVector::AddString(result, mlpack::util::GetVersion());
	});
}
void MlpackArmadilloVersion(DataChunk &args, ExpressionState &state, Vector &result) {
	Vector ignored_vector(LogicalType::VARCHAR, false, false, 1);
	UnaryExecutor::Execute<string_t, string_t>(ignored_vector, result, 1, [&](string_t unused) {
		return StringVector::AddString(result, arma_version::as_string());
	});
}


} // namespace duckdb
