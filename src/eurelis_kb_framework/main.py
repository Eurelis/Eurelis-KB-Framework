import click

from eurelis_kb_framework.output import Verbosity


@click.group()
@click.option("--verbose/--no-verbose", default=False)
@click.option("-config", default=None)
@click.pass_context
def cli(ctx, **kwargs):
    """
    Root command, will handle retrieving verbose and config options values
    Args:
        ctx: click context
        **kwargs: arguments

    Returns:

    """

    # singleton method to instantiate the LangchainWrapper
    def wrapper() -> "LangchainWrapper":
        if not wrapper.instance:
            # Get and prepare the factory
            from eurelis_kb_framework import LangchainWrapperFactory

            factory = LangchainWrapperFactory()

            if "verbose" in kwargs and kwargs.get("verbose"):
                factory.set_verbose(Verbosity.CONSOLE_DEBUG)
            else:
                factory.set_verbose(Verbosity.CONSOLE_INFO)

            if "config" in kwargs and kwargs["config"]:
                factory.set_config_path(kwargs["config"])

            wrapper.instance = factory.build(None)
        return wrapper.instance

    wrapper.instance = None
    ctx.obj["singleton"] = wrapper


@cli.group()
@click.option("--id", default=None, help="Dataset ID")
@click.pass_context
def dataset(ctx, **kwargs):
    """
    Method handling dataset options
    Args:
        ctx: click context
        **kwargs: options
    Returns:

    """
    dataset_id = kwargs["id"] if "id" in kwargs else None
    wrapper = ctx.obj["singleton"]()
    ctx.obj["wrapper"] = wrapper
    ctx.obj["dataset_id"] = dataset_id


@dataset.command("index")
@click.pass_context
def dataset_index(ctx, **kwargs):
    """
    Launch indexation
    Args:
        ctx: click context
        **kwargs: options

    Returns:

    """
    wrapper = ctx.obj["wrapper"]
    wrapper.index_documents(ctx.obj["dataset_id"])


@dataset.command("metadata")
@click.pass_context
def dataset_metadata(ctx, **kwargs):
    """
    Print first doc metadata
    Args:
        ctx: click context
        **kwargs: options

    Returns:

    """
    wrapper = ctx.obj["wrapper"]
    wrapper.print_metadata(ctx.obj["dataset_id"])


@dataset.command("cache")
@click.pass_context
def dataset_cache(ctx, **kwargs):
    """
    Write cache files for metadata
    Args:
        ctx: click context
        **kwargs: options

    Returns:

    """
    wrapper = ctx.obj["wrapper"]
    wrapper.write_files(ctx.obj["dataset_id"])


@dataset.command("ls")
@click.pass_context
def dataset_list(ctx):
    """
    List dataset
    Args:
        ctx: click context

    Returns:

    """
    wrapper = ctx.obj["wrapper"]
    wrapper.list_datasets()


@dataset.command("clear")
@click.pass_context
def dataset_clear(ctx):
    """
    Clear dataset
    Args:
        ctx: click context

    Returns:

    """
    wrapper = ctx.obj["wrapper"]
    wrapper.clear_datasets(ctx.obj["dataset_id"])


@cli.command()
@click.option("--id", default=None, help="Dataset ID")
@click.option("filters", "--filter", multiple=True, type=str)
@click.pass_context
def delete(ctx, filters, **kwargs):
    """
    Delete content from database using a query
    Args:
        ctx: click context
        filters: filters to apply to search query
        **kwargs: options

    Returns:

    """
    dataset_id = kwargs["id"] if "id" in kwargs else None
    wrapper = ctx.obj["singleton"]()

    filter_args = {}

    for filter_arg in filters:
        if not filter_arg:
            continue
        filter_split = filter_arg.split(":")
        filter_key = filter_split[0]
        filter_value = ":".join(filter_split[1:])

        filter_args[filter_key] = filter_value

    if not filter_args:  # if empty dict we consider a None value
        filter_args = None

    wrapper.delete(filter_args, dataset_id)


@cli.command()
@click.argument("query")
@click.option("filters", "--filter", multiple=True, type=str)
@click.pass_context
def search(ctx, query, filters):
    """
    Method to handle search
    Args:
        ctx: click context
        query: the text to look for
        filters: list of filters to apply

    Returns:

    """
    filter_args = {}

    for filter_arg in filters:
        if not filter_arg:
            continue
        filter_split = filter_arg.split(":")
        filter_key = filter_split[0]
        filter_value = ":".join(filter_split[1:])

        filter_args[filter_key] = filter_value

    if not filter_args:  # if empty dict we consider a None value
        filter_args = None

    wrapper = ctx.obj["singleton"]()
    wrapper.search_documents(query, for_print=True, search_filter=filter_args)


@cli.command()
@click.pass_context
def gradio(ctx):
    wrapper = ctx.obj["singleton"]()

    from eurelis_kb_framework import gradiochat

    gradiochat.define_chatbot(wrapper).launch()


# enable the dataset and search commands
if __name__ == "__main__":
    cli(obj={})


def main_cli():
    cli(obj={})
