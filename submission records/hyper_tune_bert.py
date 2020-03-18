from pylatex import Document, LongTable, MultiColumn


def basic_table():
    geometry_options = {
        "margin": "2.54cm",
        "includeheadfoot": True
    }
    doc = Document(page_numbers=True, geometry_options=geometry_options)

    batch_size_list = [8,12,16,32]
    learning_rate_list = [1e-5,2e-5,5e-5,1e-4]
    epochs_list = [2,4,8,12]

    rows = []
    for batch_size in batch_size_list:
        for learning_rate in learning_rate_list:
            for epochs in epochs_list:
                row = []
                row.append(batch_size)
                row.append(learning_rate)
                row.append(epochs)
                row.append(0)
                rows.append(row)




    # Generate data table
    with doc.create(LongTable("l l l l")) as data_table:
            data_table.add_hline()
            data_table.add_row(["batch size", "learning rate", "epochs", "F1"])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()
            for i in range(len(rows)):
                data_table.add_row(rows[i])

    doc.generate_pdf("hyper_tune_bert", clean_tex=False)
basic_table()