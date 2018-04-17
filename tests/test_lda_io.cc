#include <sstream>
#include <string>

#include <catch.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xtensor.hpp>

#include "../lda/lda.hpp"
#include "../lda/lda_io.hpp"


TEST_CASE("latent_dirichlet_allocation object can be saved and loaded")
{
    xt::xtensor<double, 2> const data = {
        {10, 8, 0, 1},
        { 7, 5, 1, 0},
        { 1, 0, 3, 0},
        { 0, 1, 5, 1},
        { 1, 0, 1, 2},
        { 1, 1, 0, 7},
    };

    latent_dirichlet_allocation::config config;

    config.topic_count = 3;
    config.doc_topic_prior = 0.456;
    config.topic_word_prior = 0.789;
    config.inner_iter_count = 12;
    config.outer_iter_count = 34;
    config.convergence_threshold = 0.567;

    latent_dirichlet_allocation lda{config};
    lda.fit(data);

    std::stringstream stream;
    save_lda(stream, lda);
    latent_dirichlet_allocation loaded_lda = load_lda(stream);

    CHECK(loaded_lda.get_config().topic_count == config.topic_count);
    CHECK(loaded_lda.get_config().doc_topic_prior == Approx(config.doc_topic_prior));
    CHECK(loaded_lda.get_config().topic_word_prior == Approx(config.topic_word_prior));
    CHECK(loaded_lda.get_config().inner_iter_count == config.inner_iter_count);
    CHECK(loaded_lda.get_config().outer_iter_count == config.outer_iter_count);
    CHECK(loaded_lda.get_config().convergence_threshold == Approx(config.convergence_threshold));

    double const topic_error = xt::amax(xt::abs(loaded_lda.topic_word_dirichlets()
                                                     - lda.topic_word_dirichlets()))();
    CHECK(topic_error < 1e-6);
}
