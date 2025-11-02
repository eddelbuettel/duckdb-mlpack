
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

}
