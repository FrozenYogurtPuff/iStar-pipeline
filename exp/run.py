from exp.main import ex

if __name__ == '__main__':
    # Actor BERT
    ex.run(config_updates={'plugins': "[]"})

    # Actor BERT + Single Plugins
    for plugin in [
        "dative_propn",
        "relcl_who",
        "actor_tag",
        "actor_dep",
        "actor_word_list",
        "actor_ner",
        "xcomp_ask",
        "be_nsubj",
        "by_sb",
    ]:
        ex.run(config_updates={
            'plugins': f"[{plugin}]",
        })

    # Actor BERT + Plugins
    ex.run(config_updates={
        'plugins': "actor_plugins",
    })
