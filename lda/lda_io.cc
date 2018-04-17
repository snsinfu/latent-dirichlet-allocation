#include <array>
#include <istream>
#include <ostream>
#include <vector>

#include <json.hpp>
#include <xtensor/xtensor.hpp>

#include "lda.hpp"
#include "lda_io.hpp"


namespace
{
    nlohmann::json xtensor_to_json(xt::xtensor<double, 2> const& tensor)
    {
        return nlohmann::json{
            {"shape", tensor.shape()},
            {"data", std::vector<double>{tensor.begin(), tensor.end()}}
        };
    }

    xt::xtensor<double, 2> xtensor_from_json(nlohmann::json const& json)
    {
        using container = xt::xtensor_container<std::vector<double>,
                                                2,
                                                xt::layout_type::row_major>;

        std::vector<double> data = json["data"];
        std::array<std::size_t, 2> shape = json["shape"];

        return container{std::move(data), std::move(shape), {}};
    }

    nlohmann::json config_to_json(latent_dirichlet_allocation::config const& config)
    {
        return nlohmann::json{
#define X(FIELD) {#FIELD, config.FIELD}
            X(topic_count),
            X(doc_topic_prior),
            X(topic_word_prior),
            X(outer_iter_count),
            X(inner_iter_count),
            X(convergence_threshold),
            X(doc_topic_prior),
#undef X
            {"topic_word_preconditions", xtensor_to_json(config.topic_word_preconditions)}
        };
    }

    latent_dirichlet_allocation::config config_from_json(nlohmann::json const& json)
    {
        latent_dirichlet_allocation::config config;

#define X(FIELD) config.FIELD = json[#FIELD]
        X(topic_count);
        X(doc_topic_prior);
        X(topic_word_prior);
        X(outer_iter_count);
        X(inner_iter_count);
        X(convergence_threshold);
        X(doc_topic_prior);
#undef X
        config.topic_word_preconditions = xtensor_from_json(json["topic_word_preconditions"]);

        return config;
    }
}

void save_lda(std::ostream& output, latent_dirichlet_allocation const& lda)
{
    output << nlohmann::json{
        {"config", config_to_json(lda.get_config())},
        {"topics", xtensor_to_json(lda.topic_word_dirichlets())}
    };
}

latent_dirichlet_allocation load_lda(std::istream& input)
{
    auto const json = nlohmann::json::parse(input);

    return latent_dirichlet_allocation{config_from_json(json["config"]),
                                       xtensor_from_json(json["topics"])};
}
