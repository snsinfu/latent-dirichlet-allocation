#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include <docopt.h>
#include <xtensor/xtensor.hpp>

#include "../lda/lda.hpp"
#include "../lda/lda_io.hpp"
#include "../tsv/tsv.hpp"


// Version of this command-line program.
static char const version[] = "lda 0.1.0";

// Usage message in the docopt[1][2] syntax.
//
// [1]: http://docopt.org/
// [2]: https://github.com/docopt/docopt.cpp
static char const usage[] = R"(
lda - Latent Dirichlet Allocation

Usage:
  lda train       [options] <doc> <model>
  lda classify    [options] <doc> <model>
  lda show-topics <model>
  lda -h

Options:
  -h --help                    Show this message
  --topics <number>            Topic count [default: 2]
  --doc-topic-prior <number>   Document-topic prior [default: 1.0]
  --topic-word-prior <number>  Topic-word prior [default: 1.0]
  --max-iter <number>          Max iteration [default: 100]
  --threshold <number>         Convergence threshold [default: 0.1]
  --preconditions <file>       Topic-word preconditioning file
)";

// Creates LDA configuration based on docopt options.
latent_dirichlet_allocation::config make_lda_config(std::map<std::string, docopt::value> const& options)
{
    latent_dirichlet_allocation::config config;

    if (auto const topics = options.at("--topics")) {
        config.topic_count = static_cast<std::size_t>(topics.asLong());
    }

    if (auto const doc_topic_prior = options.at("--doc-topic-prior")) {
        config.doc_topic_prior = std::stod(doc_topic_prior.asString());
    }

    if (auto const topic_word_prior = options.at("--topic-word-prior")) {
        config.topic_word_prior = std::stod(topic_word_prior.asString());
    }

    if (auto const max_iter = options.at("--max-iter")) {
        auto const max_iter_value = static_cast<int>(max_iter.asLong());
        config.outer_iter_count = max_iter_value;
        config.inner_iter_count = max_iter_value;
    }

    if (auto const threshold = options.at("--threshold")) {
        config.convergence_threshold = std::stod(threshold.asString());
    }

    if (auto const preconditions = options.at("--preconditions")) {
        std::ifstream preconditions_file{preconditions.asString()};
        config.topic_word_preconditions = load_tsv(preconditions_file);
    }

    return config;
}

// Trains LDA model with given document.
void train(std::map<std::string, docopt::value> const& options)
{
    latent_dirichlet_allocation lda{make_lda_config(options)};

    std::ifstream document_file{options.at("<doc>").asString()};
    auto const document = load_tsv(document_file);
    lda.fit(document);

    std::ofstream model_file{options.at("<model>").asString()};
    save_lda(model_file, lda);
}

// Classifies given document using a trained LDA model.
void classify(std::map<std::string, docopt::value> const& options)
{
    std::ifstream model_file{options.at("<model>").asString()};
    auto const lda = load_lda(model_file);

    std::ifstream document_file{options.at("<doc>").asString()};
    auto const document = load_tsv(document_file);

    save_tsv(std::cout, lda.transform(document));
}

// Prints the topic-word diciehlet parameters of a trained LDA model.
void show_topics(std::map<std::string, docopt::value> const& options)
{
    std::ifstream model_file{options.at("<model>").asString()};
    auto const lda = load_lda(model_file);

    save_tsv(std::cout, lda.topic_word_dirichlets());
}

// Analyzes docopt options and run the appropriate subcommand.
void dispatch(std::map<std::string, docopt::value> const& options)
{
    if (options.at("train").asBool()) {
        return train(options);
    }

    if (options.at("classify").asBool()) {
        return classify(options);
    }

    if (options.at("show-topics").asBool()) {
        return show_topics(options);
    }

    throw std::logic_error("unhandled subcommand");
}

int main(int argc, char** argv)
{
    try {
        dispatch(docopt::docopt(usage, {argv + 1, argv + argc}, true, version));
    } catch (std::exception const& e)  {
        std::cerr << "error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
