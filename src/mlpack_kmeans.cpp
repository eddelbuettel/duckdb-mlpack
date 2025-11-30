
#include "mlpack_adaboost.hpp"

namespace duckdb {

// mlpack adaboost accessor

void MlpackKmeansTrainTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	bool verbose = get_setting<bool>(context, "mlpack_verbose");
	bool silent = get_setting<bool>(context, "mlpack_silent");

	if (verbose)
		std::cout << "MlpackKmeansFunction\n";
	auto &resdata = const_cast<MlpackModelData &>(data_p.bind_data->Cast<MlpackModelData>());

	// if we have been called, return nothing
	if (resdata.data_returned) {
		output.SetCardinality(0);
		if (verbose)
			std::cout << "  done\n";
		return;
	}

	// Data aka explanatory variables i.e. 'features'
	arma::mat dataset = get_armadillo_matrix_transposed<double>(context, resdata.features);
	if (verbose)
		dataset.print("dataset");
	std::map<std::string, std::string> params = get_parameters(context, resdata.parameters);

	const int clusters = params.count("clusters") > 0 ? std::stoi(params["clusters"]) : 2;
	const int iterations = params.count("iterations") > 0 ? std::stoi(params["iterations"]) : 100;
	arma::Row<size_t> assignments; 					// aka predictedLables
	arma::mat centroids;

    mlpack::KMeans k(iterations); 	           			   // initialize
    k.Cluster(dataset, clusters, assignments, centroids);  // make call, filling 'assignments'

	//if (verbose)
	//	std::cout << SerializeObject<mlpack::KMeans>(k) << std::endl;
	store_vector(context, resdata.model, "assignments", serialize_vector(arma::conv_to<arma::vec>::from(assignments)));
	store_vector(context, resdata.model, "centroids_x",
	             serialize_vector(arma::conv_to<arma::vec>::from(centroids.row(0))));
	store_vector(context, resdata.model, "centroids_y",
	             serialize_vector(arma::conv_to<arma::vec>::from(centroids.row(1))));

	if (verbose) {
		assignments.print("predicted");
		centroids.print("centroids");
	}
	auto n = assignments.n_elem;
	output.SetCardinality(n);
	for (idx_t i = 0; i < n; i++) {
		output.data[0].SetValue(i, (int)assignments[i]);
	}

	resdata.data_returned = true; // mark that we have been called
}

} // namespace duckdb
