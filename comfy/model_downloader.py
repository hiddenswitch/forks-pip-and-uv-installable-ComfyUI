from __future__ import annotations

import collections
import logging
import operator
import os
import shutil
import sys
from collections.abc import Sequence, MutableSequence
from functools import reduce
from itertools import chain
from os.path import join
from pathlib import Path
from typing import List, Optional, Final, Set

import requests
import requests_cache
import tqdm
from huggingface_hub import dump_environment_info, hf_hub_download, scan_cache_dir, snapshot_download, HfFileSystem, CacheNotFound
from huggingface_hub.utils import GatedRepoError, LocalEntryNotFoundError, LocalTokenNotFoundError
from requests import Session
from safetensors import safe_open
from safetensors.torch import save_file

from .cli_args import args
from .cmd import folder_paths
from .cmd.folder_paths import add_model_folder_path, supported_pt_extensions
from .component_model.deprecation import _deprecate_method
from .component_model.files import canonicalize_path
from .interruption import InterruptProcessingException
from .model_downloader_types import CivitFile, HuggingFile, CivitModelsGetResponse, CivitFile_, Downloadable, UrlFile, FsspecFile, DownloadableFileList
from .utils import ProgressBar, comfy_tqdm

_session = Session()
_hf_fs = HfFileSystem()

logger = logging.getLogger(__name__)


from can_ada import parse as urlparse

from .component_model.uris import is_uri


def _get_hf_token():
    """Return the HF token if one is configured, otherwise ``None``.

    Unlike ``token=True`` (which raises ``LocalTokenNotFoundError`` when no
    token is stored), this returns ``None`` so that public repos can be
    accessed without authentication.
    """
    try:
        from huggingface_hub.utils._headers import get_token_to_send
        return get_token_to_send(None)
    except Exception:
        return None


def parse_hf_uri(uri: str) -> HuggingFile:
    """Parse an hf:// URI into a HuggingFile object.

    Supported formats::

        hf://repo/file
        hf://org/repo/path/to/file
        hf://datasets/org/repo/path/to/file
        hf://spaces/org/repo/path/to/file
    """
    url = urlparse(uri)
    hostname = url.hostname
    path_parts = [p for p in url.pathname.split("/") if p]

    if hostname in ("datasets", "spaces"):
        repo_type = hostname
    else:
        repo_type = "model"
        path_parts = [hostname] + path_parts

    if len(path_parts) < 2:
        raise ValueError(f"Invalid hf:// URI: {uri}")

    if len(path_parts) == 2:
        repo_id = path_parts[0]
        filename = path_parts[1]
    else:
        # with three or more parts the first two are org/repo; a repo name may hold a dot
        # (Comfy-Org/Qwen-Image-2.1, black-forest-labs/FLUX.2-dev)
        repo_id = f"{path_parts[0]}/{path_parts[1]}"
        filename = "/".join(path_parts[2:])

    return HuggingFile(repo_id=repo_id, filename=filename, repo_type=repo_type)


def get_filename_list(folder_name: str) -> Sequence[str]:
    return get_filename_list_with_downloadable(folder_name)


def get_folder_paths(*args, **kwargs):
    return folder_paths.get_folder_paths(*args, **kwargs)


def get_filename_list_with_downloadable(folder_name: str, known_files: Optional[List[Downloadable] | KnownDownloadables] = None) -> DownloadableFileList | list[str]:
    if known_files is None:
        known_files = _get_known_models_for_folder_name(folder_name)

    # Use the original folder_paths.get_filename_list if available (set by
    # apply_folder_paths_patches) to avoid infinite recursion when
    # folder_paths.get_filename_list has been replaced with this module's
    # get_filename_list.
    _get_existing = getattr(sys.modules[__name__], '_original_get_filename_list', None) or folder_paths.get_filename_list

    # workaround for lora loading issue, still needs to be investigated
    if sys.platform == "nt":
        existing = frozenset(_get_existing(folder_name))
        downloadable = frozenset() if args.disable_known_models else frozenset(str(f) for f in known_files)
        return list(map(canonicalize_path, sorted(list(existing | downloadable))))
    else:
        existing = _get_existing(folder_name)

        downloadable_files = []
        if not args.disable_known_models:
            downloadable_files = known_files

        return DownloadableFileList(existing, downloadable_files, folder_name=folder_name)


def get_full_path_or_raise(folder_name: str, filename: str, known_files: Optional[List[Downloadable] | KnownDownloadables] = None) -> str:
    res = get_or_download(folder_name, filename, known_files=known_files)
    if res is None:
        raise FileNotFoundError(f"{folder_name} does not contain {filename}")
    return res


def get_full_path(folder_name: str, filename: str) -> Optional[str]:
    return get_or_download(folder_name, filename)


def get_or_download(
    folder_name: str,
    filename: str,
    known_files: Optional[List[Downloadable] | KnownDownloadables] = None,
    *,
    offline: bool = False,
) -> Optional[str]:
    if is_uri(filename):
        url = urlparse(filename)
        if url.protocol == "hf:":
            hf_file = parse_hf_uri(filename)
            return get_or_download(folder_name, str(hf_file), known_files=[hf_file], offline=offline)
        elif url.protocol in ("http:", "https:"):
            url_file = UrlFile(filename)
            return get_or_download(folder_name, str(url_file), known_files=[url_file], offline=offline)
        else:
            fsspec_file = FsspecFile(filename)
            return get_or_download(folder_name, str(fsspec_file), known_files=[fsspec_file], offline=offline)

    if known_files is None:
        known_files = _get_known_models_for_folder_name(folder_name)

    filename = canonicalize_path(filename)
    # Use the original folder_paths.get_full_path if available (set by
    # apply_folder_paths_patches) to avoid infinite recursion.
    _get_full_path = getattr(sys.modules[__name__], '_original_get_full_path', None) or folder_paths.get_full_path
    path = _get_full_path(folder_name, filename)

    candidate_str_match = False
    candidate_filename_match = False
    candidate_alternate_filenames_match = False
    candidate_save_filename_match = False
    if path is None and not args.disable_known_models:
        try:
            # todo: should this be the first or last path?
            this_model_directory = folder_paths.get_folder_paths(folder_name)[0]
            known_file: Optional[HuggingFile | CivitFile] = None
            for candidate in known_files:
                candidate_str_match = canonicalize_path(str(candidate)) == filename
                candidate_filename_match = canonicalize_path(candidate.filename) == filename
                candidate_alternate_filenames_match = filename in list(map(canonicalize_path, candidate.alternate_filenames))
                candidate_save_filename_match = filename == canonicalize_path(candidate.save_with_filename)
                if (candidate_str_match
                        or candidate_filename_match
                        or candidate_alternate_filenames_match
                        or candidate_save_filename_match):
                    known_file = candidate
                    break
            if known_file is None:
                # Fallback to manager's model database
                from .manager_model_cache import get_model_entry, entry_to_downloadable
                entry = get_model_entry(folder_name, filename)
                if entry:
                    known_file = entry_to_downloadable(entry)
                    logger.debug(f"Found {filename} in manager database: {entry.url}")

            if known_file is None:
                # Fallback to Civitai's live search API (lazy, on-demand).
                from . import civitai_model_cache
                entry = civitai_model_cache.get_model_entry(folder_name, filename)
                if entry:
                    known_file = civitai_model_cache.entry_to_downloadable(entry, filename)
                    logger.debug(f"Found {filename} on Civitai: {entry[1]}")

            if known_file is None:
                logger.debug(f"get_or_download could not find {filename} in {folder_name}, known_files={known_files}")
                return path
            with comfy_tqdm() as watcher:
                if isinstance(known_file, HuggingFile):
                    if known_file.save_with_filename is not None:
                        linked_filename = known_file.save_with_filename
                    elif not known_file.force_save_in_repo_id and os.path.basename(known_file.filename) != known_file.filename:
                        linked_filename = os.path.basename(known_file.filename)
                    else:
                        linked_filename = known_file.filename

                    if known_file.force_save_in_repo_id or linked_filename is not None and os.path.dirname(known_file.filename) == "":
                        # if the known file has an overridden linked name, save it into a repo_id sub directory
                        # this deals with situations like
                        # jschoormans/controlnet-densepose-sdxl repo having diffusion_pytorch_model.safetensors
                        # it should be saved to controlnet-densepose-sdxl.safetensors
                        # since there are a bajillion diffusion_pytorch_model.safetensors, it should be downloaded by hf into jschoormans/controlnet-densepose-sdxl/diffusion_pytorch_model.safetensors
                        # then linked to the local folder to controlnet-densepose-sdxl.safetensors or some other canonical name
                        hf_destination_dir = os.path.join(this_model_directory, known_file.repo_id)
                    else:
                        hf_destination_dir = this_model_directory

                    # converted 16 bit files should be skipped
                    # todo: the file size should be replaced with a file hash
                    path = os.path.join(hf_destination_dir, known_file.filename)
                    try:
                        file_size = os.stat(path, follow_symlinks=True).st_size if os.path.isfile(path) else None
                    except:
                        file_size = None
                    if os.path.isfile(path) and file_size == known_file.size:
                        return path
                    # at this point, the file was not found with its candidate name
                    path = None

                    cache_hit = False
                    hf_hub_download_kwargs = dict(repo_id=known_file.repo_id,
                                  filename=known_file.filename,
                                  repo_type=known_file.repo_type,
                                  revision=known_file.revision,
                                  local_files_only=True,
                                  local_dir=hf_destination_dir if args.force_hf_local_dir_mode else None,
                                  token=_get_hf_token(),
                                                  )

                    with requests_cache.disabled():
                        try:
                            # always retrieve this from the cache if it already exists there
                            path = hf_hub_download(**hf_hub_download_kwargs)
                            logger.debug(f"hf_hub_download cache hit for {known_file.repo_id}/{known_file.filename}")
                            cache_hit = True
                        except (LocalEntryNotFoundError, LocalTokenNotFoundError):
                            if offline:
                                return None
                            try:
                                logger.debug(f"{folder_name}/{filename} is being downloaded from {known_file.repo_id}/{known_file.filename} candidate_str_match={candidate_str_match} candidate_filename_match={candidate_filename_match} candidate_alternate_filenames_match={candidate_alternate_filenames_match} candidate_save_filename_match={candidate_save_filename_match}")
                                hf_hub_download_kwargs.pop("local_files_only", None)
                                path = hf_hub_download(**hf_hub_download_kwargs)
                            except LocalTokenNotFoundError:
                                logger.debug(f"no HF token configured for {known_file.repo_id}/{known_file.filename}, skipping authenticated download")
                            except requests.exceptions.HTTPError as exc_info:
                                if getattr(exc_info.response, "status_code", None) == 401:
                                    raise GatedRepoError(f"{known_file.repo_id}/{known_file.filename}", response=exc_info.response)
                                logger.error(f"cannot download {known_file.repo_id}/{known_file.filename} from huggingface", exc_info=exc_info)
                            except IOError as exc_info:
                                logger.error(f"cannot reach huggingface {known_file.repo_id}/{known_file.filename}", exc_info=exc_info)
                            except Exception as exc_info:
                                logger.error(f"an exception occurred while downloading {known_file.repo_id}/{known_file.filename}. hf_hub_download kwargs={hf_hub_download_kwargs}", exc_info=exc_info)
                                dump_environment_info()
                                for key, value in os.environ.items():
                                    if key.startswith("HF_"):
                                        if key == "HF_TOKEN":
                                            value = "*****"
                                        logger.error("%s=%s", key, value)

                    if path is not None and known_file.convert_to_16_bit and file_size is not None and file_size != 0:
                        tensors = {}
                        with safe_open(path, framework="pt") as f:
                            with tqdm.tqdm(total=len(f.keys())) as pb:
                                for k in f.keys():
                                    x = f.get_tensor(k)
                                    tensors[k] = x.half()
                                    del x
                                    pb.update()

                        # always save converted files to the destination so that the huggingface cache is not corrupted
                        save_file(tensors, os.path.join(hf_destination_dir, known_file.filename))

                        for _, v in tensors.items():
                            del v
                        logger.info(f"Converted {path} to 16 bit, size is {os.stat(path, follow_symlinks=True).st_size}")

                    link_successful = True
                    exc_info_link = {}
                    if path is not None:
                        if offline:
                            return path
                        if Path(linked_filename).is_absolute():
                            raise ValueError(f"{known_file.repo_id}/{known_file.filename} surprisingly was trying to link to an absolute path {linked_filename}, failing")

                        destination_link = Path(this_model_directory) / linked_filename
                        if destination_link.is_file():
                            logger.warning(f"{known_file.repo_id}/{known_file.filename} could not link to {destination_link} because the path already exists, which is unexpected")
                        else:
                            try:
                                # sometimes, linked filename has a path in it, on purpose, such as with controlnet_aux nodes
                                Path(destination_link).parent.mkdir(parents=True, exist_ok=True)
                                os.symlink(path, destination_link)
                            except FileExistsError:
                                # the download was resumed
                                pass
                            except Exception as exc_info:
                                exc_info_link = exc_info
                                logger.error("error while symbolic linking", exc_info=exc_info)
                                try:
                                    os.link(path, destination_link)
                                except Exception as hard_link_exc:
                                    logger.error("error while hard linking", exc_info=hard_link_exc)
                                    if cache_hit:
                                        shutil.copyfile(path, destination_link)
                                    link_successful = False
                                    exc_info_link = (exc_info, hard_link_exc)

                    if not link_successful:
                        logger.error(f"Failed to link file with alternative download save name in a way that is compatible with Hugging Face caching {repr(known_file)}. If cache_hit={cache_hit} is True, the file was copied into the destination. exc_info={exc_info_link}")

                    # Download companion files (e.g. ONNX external data files)
                    if path is not None and hasattr(known_file, 'companion_files') and known_file.companion_files:
                        for comp_filename in known_file.companion_files:
                            comp_save = os.path.join(os.path.dirname(linked_filename), os.path.basename(comp_filename))
                            comp_dest = Path(this_model_directory) / comp_save
                            if comp_dest.is_file():
                                continue
                            try:
                                comp_path = hf_hub_download(
                                    repo_id=known_file.repo_id,
                                    filename=comp_filename,
                                    repo_type=known_file.repo_type,
                                    revision=known_file.revision,
                                    local_dir=hf_destination_dir if args.force_hf_local_dir_mode else None,
                                    token=_get_hf_token(),
                                )
                                comp_dest.parent.mkdir(parents=True, exist_ok=True)
                                try:
                                    os.symlink(comp_path, comp_dest)
                                except FileExistsError:
                                    pass
                                except OSError:
                                    try:
                                        os.link(comp_path, comp_dest)
                                    except OSError:
                                        shutil.copyfile(comp_path, str(comp_dest))
                                logger.info(f"Downloaded companion file {comp_filename} for {known_file.filename}")
                            except Exception as comp_exc:
                                logger.warning(f"Failed to download companion file {comp_filename}: {comp_exc}")
                else:
                    if offline:
                        return path
                    save_filename = known_file.save_with_filename or known_file.filename
                    destination_with_filename = join(this_model_directory, save_filename)
                    os.makedirs(os.path.dirname(destination_with_filename), exist_ok=True)

                    if isinstance(known_file, FsspecFile):
                        # Handle FsspecFile using fsspec
                        import fsspec
                        try:
                            with fsspec.open(known_file.uri, "rb") as src:
                                with open(destination_with_filename, "wb") as dst:
                                    chunk_size = 1024 * 1024  # 1MB
                                    while True:
                                        chunk = src.read(chunk_size)
                                        if not chunk:
                                            break
                                        dst.write(chunk)
                        except InterruptProcessingException:
                            os.remove(destination_with_filename)

                        path = folder_paths.get_full_path(folder_name, filename)
                        assert path is not None
                    else:
                        # Handle URL-based downloads (CivitFile, UrlFile)
                        url: Optional[str] = None

                        if isinstance(known_file, CivitFile):
                            model_info_res = _session.get(
                                f"https://civitai.com/api/v1/models/{known_file.model_id}?modelVersionId={known_file.model_version_id}")
                            model_info: CivitModelsGetResponse = model_info_res.json()

                            civit_file: CivitFile_
                            for civit_file in chain.from_iterable(version['files'] for version in model_info.get('modelVersions', [])):
                                if canonicalize_path(civit_file['name']) == filename:
                                    url = civit_file['downloadUrl']
                                    break
                        elif isinstance(known_file, UrlFile):
                            url = known_file.url
                        else:
                            raise RuntimeError("Unknown file type")

                        if url is None:
                            logger.warning(f"Could not retrieve file {str(known_file)}")
                        else:
                            try:
                                with _session.get(url, stream=True, allow_redirects=True) as response:
                                    total_size = int(response.headers.get("content-length", 0))
                                    progress_bar = ProgressBar(total=total_size)
                                    with open(destination_with_filename, "wb") as file:
                                        for chunk in response.iter_content(chunk_size=512 * 1024):
                                            progress_bar.update(len(chunk))
                                            file.write(chunk)
                            except InterruptProcessingException:
                                os.remove(destination_with_filename)

                            path = folder_paths.get_full_path(folder_name, filename)
                            assert path is not None
        except StopIteration:
            pass
        except GatedRepoError as exc_info:
            exc_info.append_to_message(f"""
Visit the repository, accept the terms, and then do one of the following:

 - Set the HF_TOKEN environment variable to your Hugging Face token; or,
 - Login to Hugging Face in your terminal using `huggingface-cli login`
""")
            raise exc_info
    return path


class KnownDownloadables(collections.UserList[Downloadable]):
    # we're not invoking the constructor because we want a reference to the passed list
    # noinspection PyMissingConstructor
    def __init__(self, data, folder_name: Optional[str | Sequence[str]] = None, folder_names: Optional[Sequence[str]] = None):
        # this should be a view
        self.data = data
        folder_names = folder_names or []
        if isinstance(folder_name, str):
            folder_names.append(folder_name)
        elif folder_name is not None and hasattr(folder_name, "__getitem__") and len(folder_name[0]) > 1:
            folder_names += folder_name
        self._folder_names = folder_names

    @property
    def folder_names(self) -> list[str]:
        return self._folder_names

    @folder_names.setter
    def folder_names(self, value: list[str]):
        self._folder_names = value

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self._folder_names
        else:
            return item in self.data


KNOWN_CHECKPOINTS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Comfy-Org/YuE2", "checkpoints/yue2_3b_bf16.safetensors"),
    HuggingFile("Comfy-Org/YuE2", "checkpoints/yue2_3b_int8_convrot.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors", alternate_filenames=("SDXL/sd_xl_base_1.0.safetensors", "SDXL/sd_xl_base_1.0_0.9vae.safetensors", "sdxl/sd_xl_base_1.0.safetensors", "xl/Colv49-Protov66-JugX-32-32-35.safetensors")),
    HuggingFile("stabilityai/stable-diffusion-xl-refiner-1.0", "sd_xl_refiner_1.0.safetensors", alternate_filenames=("SDXL/sd_xl_refiner_1.0.safetensors", "SDXL/sd_xl_refiner_1.0_0.9vae.safetensors", "sdxl/sd_xl_refiner_1.0.safetensors")),
    HuggingFile("stabilityai/sdxl-turbo", "sd_xl_turbo_1.0_fp16.safetensors"),
    HuggingFile("stabilityai/sdxl-turbo", "sd_xl_turbo_1.0.safetensors", show_in_ui=False),
    HuggingFile("stabilityai/stable-cascade", "comfyui_checkpoints/stable_cascade_stage_b.safetensors"),
    HuggingFile("stabilityai/stable-cascade", "comfyui_checkpoints/stable_cascade_stage_c.safetensors"),
    HuggingFile("stabilityai/stable-cascade", "comfyui_checkpoints/stage_a.safetensors", show_in_ui=False),
    HuggingFile("Comfy-Org/stable-diffusion-v1-5-archive", "v1-5-pruned-emaonly.safetensors", alternate_filenames=("main/v1-5-pruned-emaonly.safetensors", "sd15/v1-5-pruned-emaonly.safetensors")),
    HuggingFile("Comfy-Org/stable-diffusion-v1-5-archive", "v1-5-pruned-emaonly-fp16.safetensors", alternate_filenames=("main/v1-5-pruned-emaonly-fp16.safetensors", "sd15/v1-5-pruned-emaonly-fp16.safetensors")),
    # from https://github.com/comfyanonymous/ComfyUI_examples/tree/master/2_pass_txt2img
    HuggingFile("stabilityai/stable-diffusion-2-1", "v2-1_768-ema-pruned.ckpt", show_in_ui=False),
    HuggingFile("waifu-diffusion/wd-1-5-beta3", "wd-illusion-fp16.safetensors", show_in_ui=False),
    HuggingFile("jomcs/NeverEnding_Dream-Feb19-2023", "CarDos Anime/cardosAnime_v10.safetensors", show_in_ui=False),
    # from https://github.com/comfyanonymous/ComfyUI_examples/blob/master/area_composition/README.md
    HuggingFile("ckpt/anything-v3.0", "Anything-V3.0.ckpt", show_in_ui=False),
    HuggingFile("stabilityai/cosxl", "cosxl.safetensors", alternate_filenames=("cosxl/cosxl.safetensors",)),
    HuggingFile("stabilityai/cosxl", "cosxl_edit.safetensors", alternate_filenames=("cosxl/cosxl_edit.safetensors",)),
    # latest, popular civitai models
    CivitFile(133005, 357609, filename="juggernautXL_v9Rundiffusionphoto2.safetensors", alternate_filenames=("_SDXL_/juggernautXL_v9Rundiffusionphoto2.safetensors", "sdxl/juggernautXL_v9Rundiffusionphoto2.safetensors")),
    CivitFile(112902, 351306, filename="dreamshaperXL_v21TurboDPMSDE.safetensors"),
    CivitFile(139562, 344487, filename="realvisxlV40_v40Bakedvae.safetensors"),
    HuggingFile("SG161222/Realistic_Vision_V6.0_B1_noVAE", "Realistic_Vision_V6.0_NV_B1_fp16.safetensors"),
    HuggingFile("SG161222/Realistic_Vision_V5.1_noVAE", "Realistic_Vision_V5.1_fp16-no-ema.safetensors", alternate_filenames=("sd15/Realistic_Vision_V5.1_fp16-no-ema.safetensors", "SD1.5/fantexiRealistic_v10.safetensors", "SD1.5/noosphere_v42.safetensors", "SD1.5/V07_v07.safetensors")),
    HuggingFile("Lykon/DreamShaper", "DreamShaper_8_pruned.safetensors", save_with_filename="dreamshaper_8.safetensors", alternate_filenames=("DreamShaper_8_pruned.safetensors", "sd1.5/dreamshaper_8.safetensors")),
    CivitFile(7371, 425083, filename="revAnimated_v2Rebirth.safetensors"),
    CivitFile(4468, 57618, filename="counterfeitV30_v30.safetensors"),
    CivitFile(241415, 272376, filename="picxReal_10.safetensors"),
    CivitFile(23900, 95489, filename="anyloraCheckpoint_bakedvaeBlessedFp16.safetensors"),
    CivitFile(132803, 146134, filename="fantexiRealistic_v10.safetensors"),
    CivitFile(36538, 265285, filename="noosphere_v42.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-3-medium", "sd3_medium.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-3-medium", "sd3_medium_incl_clips.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-3-medium", "sd3_medium_incl_clips_t5xxlfp8.safetensors"),
    HuggingFile("fal/AuraFlow", "aura_flow_0.1.safetensors"),
    # stable audio, # uses names from https://comfyanonymous.github.io/ComfyUI_examples/audio/
    HuggingFile("Comfy-Org/stable-audio-open-1.0_repackaged", "stable-audio-open-1.0.safetensors", alternate_filenames=("stable_audio_open_1.0.safetensors",)),
    # hunyuandit
    HuggingFile("comfyanonymous/hunyuan_dit_comfyui", "hunyuan_dit_1.0.safetensors"),
    HuggingFile("comfyanonymous/hunyuan_dit_comfyui", "hunyuan_dit_1.1.safetensors"),
    HuggingFile("comfyanonymous/hunyuan_dit_comfyui", "hunyuan_dit_1.2.safetensors"),
    HuggingFile("lllyasviel/flux1-dev-bnb-nf4", "flux1-dev-bnb-nf4.safetensors"),
    HuggingFile("lllyasviel/flux1-dev-bnb-nf4", "flux1-dev-bnb-nf4-v2.safetensors"),
    HuggingFile("silveroxides/flux1-nf4-weights", "flux1-schnell-bnb-nf4.safetensors"),
    HuggingFile("Lightricks/LTX-Video", "ltx-video-2b-v0.9.safetensors"),
    HuggingFile("Lightricks/LTX-Video", "ltx-video-2b-v0.9.1.safetensors"),
    HuggingFile("Lightricks/LTX-2", "ltx-2-19b-dev-fp8.safetensors"),
    HuggingFile("Lightricks/LTX-2", "ltx-2-19b-dev.safetensors"),
    HuggingFile("Lightricks/LTX-2.3-fp8", "ltx-2.3-22b-dev-fp8.safetensors"),
    HuggingFile("Lightricks/LTX-2.3-fp8", "ltx-2.3-22b-distilled-fp8.safetensors"),
    HuggingFile("Lightricks/LTX-2.3", "ltx-2.3-22b-dev.safetensors"),
    HuggingFile("Lightricks/LTX-2.3", "ltx-2.3-22b-distilled.safetensors"),
    HuggingFile("Comfy-Org/Lumina_Image_2.0_Repackaged", "all_in_one/lumina_2.safetensors"),
    HuggingFile("Comfy-Org/flux1-schnell", "flux1-schnell-fp8.safetensors"),
    HuggingFile("Comfy-Org/flux1-dev", "flux1-dev-fp8.safetensors"),
    HuggingFile("stabilityai/stable-video-diffusion-img2vid", "svd.safetensors"),
    HuggingFile("stabilityai/stable-video-diffusion-img2vid-xt", "svd_xt.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-2-inpainting", "v2-inpainting-pruned-ema.safetensors"),
    HuggingFile("runwayml/stable-diffusion-inpainting", "sd-v1-5-inpainting.ckpt", show_in_ui=False),
    HuggingFile("stabilityai/stable-diffusion-3.5-large", "sd3.5_large.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-3.5-medium", "sd3.5_medium.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-3.5-large-turbo", "sd3.5_large_turbo.safetensors"),
    HuggingFile("Comfy-Org/stable-diffusion-3.5-fp8", "sd3.5_large_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/stable-diffusion-3.5-fp8", "sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors"),
    HuggingFile("fal/AuraFlow-v0.2", "aura_flow_0.2.safetensors"),
    HuggingFile("lodestones/Chroma1-Base", "Chroma1-Base.safetensors"),
    HuggingFile("Comfy-Org/mochi_preview_repackaged", "all_in_one/mochi_preview_fp8_scaled.safetensors"),
    HuggingFile("Lightricks/LTX-Video", "ltx-video-2b-v0.9.5.safetensors"),
    HuggingFile("Comfy-Org/ace_step_1.5_ComfyUI_files", "checkpoints/ace_step_1.5_turbo_aio.safetensors"),
    HuggingFile("Comfy-Org/ACE-Step_ComfyUI_repackaged", "all_in_one/ace_step_v1_3.5b.safetensors"),
    HuggingFile("Comfy-Org/SDPose", "checkpoints/sdpose_wholebody_fp16.safetensors"),
    HuggingFile("Comfy-Org/sam3.1", "checkpoints/sam3.1_multiplex_fp16.safetensors"),
    CivitFile(8714, 13359, filename="AOM2-Hard.safetensors"),
    CivitFile(4291, 132454, filename="AOM3A3.safetensors"),
    CivitFile(140737, 357037, filename="albedobaseXL_v21.safetensors", alternate_filenames=("sdxl/AlbedoBaseXL.safetensors", "sdxl/albedobaseXL_v21.safetensors")),
], folder_name="checkpoints")

KNOWN_UNCLIP_CHECKPOINTS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("stabilityai/stable-cascade", "comfyui_checkpoints/stable_cascade_stage_c.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-2-1-unclip", "sd21-unclip-h.ckpt"),
    HuggingFile("stabilityai/stable-diffusion-2-1-unclip", "sd21-unclip-l.ckpt"),
    HuggingFile("comfyanonymous/wd-1.5-beta2_unCLIP", "wd-1-5-beta2-aesthetic-unclip-h.safetensors"),
    HuggingFile("comfyanonymous/illuminatiDiffusionV1_v11_unCLIP", "illuminatiDiffusionV1_v11-unclip-h.safetensors"),
], folder_name="checkpoints")

KNOWN_IMAGE_ONLY_CHECKPOINTS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("stabilityai/stable-zero123", "stable_zero123.ckpt")
], folder_name="checkpoints")

KNOWN_UPSCALERS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("lllyasviel/Annotators", "RealESRGAN_x4plus.pth"),
    HuggingFile("Kallamamran/best_upscaler_models", "4xNomos8k_atd_jpg.pth"),
    HuggingFile("Comfy-Org/Real-ESRGAN_repackaged", "RealESRGAN_x4plus.safetensors"),
], folder_name="upscale_models")

KNOWN_LATENT_UPSCALE_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Lightricks/LTX-2", "ltx-2-spatial-upscaler-x2-1.0.safetensors"),
    HuggingFile("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
    HuggingFile("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
    HuggingFile("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors"),
    HuggingFile("Lightricks/LTX-2.3", "ltx-2.3-temporal-upscaler-x2-1.0.safetensors"),
    HuggingFile("Lightricks/LTX-2.5", "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
    HuggingFile("Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/latent_upscale_models/hunyuanvideo15_latent_upsampler_1080p.safetensors"),
], folder_name="latent_upscale_models")

KNOWN_GLIGEN_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("comfyanonymous/GLIGEN_pruned_safetensors", "gligen_sd14_textbox_pruned.safetensors", show_in_ui=False),
    HuggingFile("comfyanonymous/GLIGEN_pruned_safetensors", "gligen_sd14_textbox_pruned_fp16.safetensors"),
], folder_name="gligen")

KNOWN_CLIP_VISION_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("comfyanonymous/clip_vision_g", "clip_vision_g.safetensors"),
    HuggingFile("Comfy-Org/sigclip_vision_384", "sigclip_vision_patch14_384.safetensors", alternate_filenames=("main/sigclip_vision_patch14_384.safetensors",)),
    HuggingFile("Comfy-Org/HunyuanVideo_repackaged", "split_files/clip_vision/llava_llama3_vision.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/clip_vision/clip_vision_h.safetensors"),
    HuggingFile("Comfy-Org/CLIP-ViT-H-14-laion2B-s32B-b79K_repackaged", "split_files/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"),
    HuggingFile("VAST-AI/TripoSplat", "clip_vision/dino_v3_vit_h.safetensors"),
    HuggingFile("Comfy-Org/Pixal3D", "clip_vision/dino_v3_L_naf_fp32.safetensors"),
    # WanVideoWrapper (Kijai) -- CLIP vision
    HuggingFile("Kijai/WanVideo_comfy", "open-clip-xlm-roberta-large-vit-huge-14_visual_fp16.safetensors", show_in_ui=False),
], folder_name="clip_vision")

KNOWN_EMBEDDINGS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Comfy-Org/marigold-v2-0", "embeddings/marigold_v2_albedo_conditioning.safetensors"),
    HuggingFile("Comfy-Org/marigold-v2-0", "embeddings/marigold_v2_depth_conditioning.safetensors"),
    HuggingFile("Comfy-Org/marigold-v2-0", "embeddings/marigold_v2_normals_conditioning.safetensors"),
], folder_name="embeddings")

KNOWN_LORAS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Comfy-Org/marigold-v2-0", "loras/marigold_v2_albedo.safetensors"),
    HuggingFile("Comfy-Org/marigold-v2-0", "loras/marigold_v2_depth_log_stage2.safetensors"),
    HuggingFile("Comfy-Org/marigold-v2-0", "loras/marigold_v2_normals.safetensors"),
    CivitFile(model_id=211577, model_version_id=238349, filename="openxl_handsfix.safetensors"),
    CivitFile(model_id=324815, model_version_id=364137, filename="blur_control_xl_v1.safetensors"),
    CivitFile(model_id=47085, model_version_id=55199, filename="GoodHands-beta2.safetensors"),
    HuggingFile("artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5", "PixelArtRedmond15V-PixelArt-PIXARFK.safetensors"),
    HuggingFile("ByteDance/Hyper-SD", "Hyper-SDXL-12steps-CFG-lora.safetensors"),
    HuggingFile("ByteDance/Hyper-SD", "Hyper-SD15-12steps-CFG-lora.safetensors"),
    HuggingFile("black-forest-labs/FLUX.1-Canny-dev-lora", "flux1-canny-dev-lora.safetensors"),
    HuggingFile("black-forest-labs/FLUX.1-Depth-dev-lora", "flux1-depth-dev-lora.safetensors"),
    HuggingFile("latent-consistency/lcm-lora-sdxl", "pytorch_lora_weights.safetensors", save_with_filename="lcm_lora_sdxl.safetensors"),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Lightning-4steps-V1.0.safetensors", show_in_ui=False),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors", show_in_ui=False),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Lightning-4steps-V2.0.safetensors"),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors", show_in_ui=False),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Lightning-8steps-V1.0.safetensors", show_in_ui=False),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Lightning-8steps-V1.1.safetensors", show_in_ui=False),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors", show_in_ui=False),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Lightning-8steps-V2.0.safetensors"),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors"),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Edit-Lightning-4steps-V1.0.safetensors"),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors", show_in_ui=False),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors"),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors", show_in_ui=False),
    HuggingFile("Lightricks/LTX-2", "ltx-2-19b-distilled-lora-384.safetensors"),
    HuggingFile("Lightricks/LTX-2.3", "ltx-2.3-22b-distilled-lora-384.safetensors"),
    HuggingFile("Comfy-Org/Krea-2", "loras/krea2_darkbrush.safetensors"),
    HuggingFile("Comfy-Org/Krea-2", "loras/krea2_style_reference.safetensors"),
    HuggingFile("Comfy-Org/SCAIL-2", "loras/wan2.1_SCAIL_2_DPO_lora_bf16.safetensors"),
    HuggingFile("circlestone-labs/Anima-Official-LoRAs", "anima-turbo-lora-v0.2.safetensors"),
    HuggingFile("lightx2v/Minimax-h3-Turbo", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),
    HuggingFile("Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left", "ltx-2-19b-lora-camera-control-dolly-left.safetensors"),
    HuggingFile("Comfy-Org/flux2-dev", "split_files/loras/Flux2TurboComfyv2.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/loras/Qwen-Edit-2509-Multiple-angles.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/loras/Qwen-Image-Edit-2509-Anything2RealAlpha.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/loras/Qwen-Image-Edit-2509-Fusion.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/loras/Qwen-Image-Edit-2509-Light-Migration.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/loras/Qwen-Image-Edit-2509-Relight.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/loras/chronoedit_distill_lora.safetensors"),
    HuggingFile("Comfy-Org/ltx-2", "split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors"),
    HuggingFile("Comfy-Org/ltx-2", "split_files/loras/ltx2-squish.safetensors"),
    HuggingFile("Comfy-Org/ltx-2.3", "split_files/loras/ltx-2.3-id-lora-talkvid-3k.safetensors"),
    HuggingFile("Comfy-Org/ltx-2.3", "split_files/loras/ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"),
    HuggingFile("Comfy-Org/ltx-2.3", "split_files/loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-DiffSynth-ControlNets", "split_files/loras/qwen_image_union_diffsynth_lora.safetensors"),
    HuggingFile("Comfy-Org/USO_1.0_Repackaged", "split_files/loras/uso-flux1-dit-lora-v1.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/loras/wan_alpha_2.1_rgba_lora.safetensors"),
    # WanVideoWrapper (Kijai) -- LoRAs
    HuggingFile("Kijai/WanVideo_comfy", "Lightx2v/lightx2v_14B_T2V_cfg_step_distill_lora_adaptive_rank_quantile_0.15_bf16.safetensors", save_with_filename="WanVideo/Lightx2v/lightx2v_14B_T2V_cfg_step_distill_lora_adaptive_rank_quantile_0.15_bf16.safetensors", alternate_filenames=("lightx2v_14B_T2V_cfg_step_distill_lora_adaptive_rank_quantile_0.15_bf16.safetensors",), show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", save_with_filename="WanVideo/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", alternate_filenames=("WanVid/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"), show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank32_bf16.safetensors", save_with_filename="WanVideo/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank32_bf16.safetensors", alternate_filenames=("WanVideo/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank32_bf16.safetensors", "lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank32_bf16.safetensors"), show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors", save_with_filename="WanVideo/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors", alternate_filenames=("WanVideo/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors", "lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors"), show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "LoRAs/Wan21_T2V_14B/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors", save_with_filename="Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "Wan21_CausVid_14B_T2V_lora_rank32.safetensors", save_with_filename="WanVideo/Wan21_CausVid_14B_T2V_lora_rank32.safetensors", alternate_filenames=("Wan21_CausVid_14B_T2V_lora_rank32.safetensors",), show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors", save_with_filename="WanVideo/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors", alternate_filenames=("Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors",), show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "LoRAs/CausVid/Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors", save_with_filename="Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "LoRAs/Wan2_1_self_forcing_1_3B/Wan2_1_self_forcing_dmd_1_3B_lora_rank_32_fp16.safetensors", save_with_filename="Wan2_1_self_forcing_dmd_1_3B_lora_rank_32_fp16.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "Pusa/Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors", save_with_filename="WanVideo/Pusa/Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "Pusa/Wan22_PusaV1_lora_HIGH_resized_dynamic_avg_rank_98_bf16.safetensors", save_with_filename="WanVideo/Pusa/Wan22_PusaV1_lora_HIGH_resized_dynamic_avg_rank_98_bf16.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "Pusa/Wan22_PusaV1_lora_LOW_resized_dynamic_avg_rank_98_bf16.safetensors", save_with_filename="WanVideo/Pusa/Wan22_PusaV1_lora_LOW_resized_dynamic_avg_rank_98_bf16.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors", save_with_filename="WanVideo/Wan22-Lightning/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors", alternate_filenames=("Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors",), show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors", save_with_filename="WanVideo/Wan22-Lightning/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors", alternate_filenames=("Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors",), show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_T2V-A14B-4steps-lora_HIGH_fp16.safetensors", save_with_filename="WanVideo/Wan22-Lightning/Wan2.2-Lightning_T2V-A14B-4steps-lora_HIGH_fp16.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "LoRAs/Stand-In/Stand-In_wan2.1_T2V_14B_ver1.0_fp16.safetensors", save_with_filename="WanVideo/Stand-In/Stand-In_wan2.1_T2V_14B_ver1.0.safetensors", alternate_filenames=("WanVideo/Stand-In/Stand-In_wan2.1_T2V_14B_ver1.0_fp16.safetensors",), show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors", save_with_filename="WanVideo/WanAnimate_relight_lora_fp16.safetensors", alternate_filenames=("WanAnimate_relight_lora_fp16.safetensors",), show_in_ui=False),
    HuggingFile("Kijai/LongCat-Video_comfy", "LongCat_distill_lora_alpha64_bf16.safetensors", save_with_filename="LongCat_distill_lora_alpha64_bf16.safetensors", alternate_filenames=("LongCat_distill_lora_rank128_bf16.safetensors",), show_in_ui=False),
    # WanVideoWrapper -- reward LoRAs
    HuggingFile("Kijai/Wan2.1-Fun-Reward-LoRAs-comfy", "Wan2.1-Fun-1.3B-InP-MPS_reward_lora_comfy.safetensors", save_with_filename="WanVid/funreward/Wan2.1-Fun-1.3B-InP-MPS_reward_lora_comfy.safetensors", show_in_ui=False),
    # WanVideoWrapper -- control LoRAs
    HuggingFile("spacepxl/Wan2.1-control-loras", "1.3b/tile/wan2.1-1.3b-control-lora-tile-v1.1_comfy.safetensors", save_with_filename="WanVid/wan2.1-1.3b-control-lora-tile-v1.1_comfy.safetensors", alternate_filenames=("WanVid/wan2.1-1.3b-control-lora-tile-v0.1_comfy.safetensors", "WanVid\\wan2.1-1.3b-control-lora-tile-v0.1_comfy.safetensors"), show_in_ui=False),
    # LeapFusion HunyuanVideo i2v LoRA
    HuggingFile("leapfusion-image2vid-test/image2vid-960x544", "img2vid544p.safetensors", save_with_filename="hyvid/musubi-tuner/img2vid544p.safetensors", show_in_ui=False),
    # --- Models referenced by comfyui-workflow-templates (sourced from
    # MarkdownNote links inside each template's UI JSON). Keep this
    # section's coverage complete against Comfy-Org/workflow_templates;
    # a scan diffs widgets_values against this DB in tests.
    # Qwen-Image / Qwen-Image-Edit Lightning acceleration LoRAs (lightx2v)
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors", save_with_filename="Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"),
    HuggingFile("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors", save_with_filename="Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors"),
    HuggingFile("lightx2v/Qwen-Image-Edit-2511-Lightning", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"),
    HuggingFile("lightx2v/Qwen-Image-2512-Lightning", "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors"),
    # Qwen-Image multi-angle and style LoRAs
    HuggingFile("fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA", "qwen-image-edit-2511-multiple-angles-lora.safetensors"),
    HuggingFile("alvdansen/illustration-1.0-qwen-image", "illustration-1.0-qwen-image.safetensors"),
    HuggingFile("enigmatic/gummycandy_qwen", "gummycandy_qwen.safetensors"),
    # Wuli Qwen-Image-2512 Turbo 2-step LoRA
    HuggingFile("Wuli-art/Qwen-Image-2512-Turbo-LoRA-2-Steps", "Wuli-Qwen-Image-2512-Turbo-LoRA-2steps-V1.0-bf16.safetensors"),
    # Flux.2 Turbo LoRA
    HuggingFile("ByteZSzn/Flux.2-Turbo-ComfyUI", "Flux_2-Turbo-LoRA_comfyui.safetensors"),
    # LTX-2 IC-LoRA control packs
    HuggingFile("Lightricks/LTX-2-19b-IC-LoRA-Canny-Control", "ltx-2-19b-ic-lora-canny-control.safetensors"),
    HuggingFile("Lightricks/LTX-2-19b-IC-LoRA-Depth-Control", "ltx-2-19b-ic-lora-depth-control.safetensors"),
    HuggingFile("Lightricks/LTX-2-19b-IC-LoRA-Pose-Control", "ltx-2-19b-ic-lora-pose-control.safetensors"),
    HuggingFile("valiantcat/LTX-2.3-Transition-LORA", "ltx2.3-transition.safetensors"),
    # Lightx2v WanVideo acceleration LoRA (Kijai repack)
    HuggingFile("Kijai/WanVideo_comfy", "Lightx2v/lightx2v_14B_T2V_cfg_step_distill_lora_adaptive_rank_quantile_0.15_bf16.safetensors"),
], folder_name="loras")

KNOWN_MODEL_PATCHES: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Comfy-Org/MiniMax-H3", "model_patches/minimax_h3_fun_controlnet_union_pruned_bf16.safetensors"),
    HuggingFile("Comfy-Org/MiniMax-H3", "model_patches/minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors"),
], folder_name="model_patches")

KNOWN_CONTROLNETS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("beycanai/ControlNet-models-INT8-ConvRot", "Qwen-Image-2512-Fun-Controlnet-Union-2602_int8_convrot.safetensors"),
    HuggingFile("beycanai/ControlNet-models-INT8-ConvRot", "Qwen-Image-InstantX-ControlNet-Union_int8_convrot.safetensors"),
    HuggingFile("beycanai/ControlNet-models-INT8-ConvRot", "Z-Image-Turbo-Fun-Controlnet-Union-2.1_int8_convrot.safetensors"),
    HuggingFile("thibaud/controlnet-openpose-sdxl-1.0", "OpenPoseXL2.safetensors", convert_to_16_bit=True, size=2502139104),
    HuggingFile("thibaud/controlnet-openpose-sdxl-1.0", "control-lora-openposeXL2-rank256.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11e_sd15_ip2p_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11e_sd15_shuffle_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11f1p_sd15_depth_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11p_sd15_canny_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11p_sd15_inpaint_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11p_sd15_lineart_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11p_sd15_mlsd_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11p_sd15_normalbae_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11p_sd15_openpose_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11p_sd15_scribble_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11p_sd15_seg_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11p_sd15_softedge_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_lora_rank128_v11p_sd15s2_lineart_anime_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11e_sd15_ip2p_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11e_sd15_shuffle_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11f1e_sd15_tile_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11f1p_sd15_depth_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11p_sd15_canny_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11p_sd15_inpaint_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11p_sd15_lineart_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11p_sd15_mlsd_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11p_sd15_normalbae_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11p_sd15_openpose_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11p_sd15_scribble_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11p_sd15_seg_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11p_sd15_softedge_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11p_sd15s2_lineart_anime_fp16.safetensors"),
    HuggingFile("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11u_sd15_tile_fp16.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "diffusers_xl_canny_full.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "diffusers_xl_canny_mid.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "diffusers_xl_canny_small.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "diffusers_xl_depth_full.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "diffusers_xl_depth_mid.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "diffusers_xl_depth_small.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "ioclab_sd15_recolor.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "kohya_controllllite_xl_blur.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "kohya_controllllite_xl_blur_anime.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "kohya_controllllite_xl_blur_anime_beta.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "kohya_controllllite_xl_canny.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "kohya_controllllite_xl_canny_anime.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "kohya_controllllite_xl_depth.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "kohya_controllllite_xl_depth_anime.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "kohya_controllllite_xl_openpose_anime.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "kohya_controllllite_xl_openpose_anime_v2.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "kohya_controllllite_xl_scribble_anime.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sai_xl_canny_128lora.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sai_xl_canny_256lora.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sai_xl_depth_128lora.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sai_xl_depth_256lora.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sai_xl_recolor_128lora.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sai_xl_recolor_256lora.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sai_xl_sketch_128lora.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sai_xl_sketch_256lora.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sargezt_xl_depth.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sargezt_xl_depth_faid_vidit.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sargezt_xl_depth_zeed.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "sargezt_xl_softedge.safetensors"),
    HuggingFile("SargeZT/controlnet-sd-xl-1.0-depth-16bit-zoe", "depth-zoe-xl-v1.0-controlnet.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "t2i-adapter_diffusers_xl_canny.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "t2i-adapter_diffusers_xl_depth_midas.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "t2i-adapter_diffusers_xl_depth_zoe.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "t2i-adapter_diffusers_xl_lineart.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "t2i-adapter_diffusers_xl_openpose.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "t2i-adapter_diffusers_xl_sketch.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "t2i-adapter_xl_canny.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "t2i-adapter_xl_openpose.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "t2i-adapter_xl_sketch.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "thibaud_xl_openpose.safetensors"),
    HuggingFile("lllyasviel/sd_control_collection", "thibaud_xl_openpose_256lora.safetensors"),
    HuggingFile("jschoormans/controlnet-densepose-sdxl", "diffusion_pytorch_model.safetensors", save_with_filename="controlnet-densepose-sdxl.safetensors", convert_to_16_bit=True, size=2502139104),
    HuggingFile("stabilityai/stable-cascade", "controlnet/canny.safetensors", save_with_filename="stable_cascade_canny.safetensors"),
    HuggingFile("stabilityai/stable-cascade", "controlnet/inpainting.safetensors", save_with_filename="stable_cascade_inpainting.safetensors"),
    HuggingFile("stabilityai/stable-cascade", "controlnet/super_resolution.safetensors", save_with_filename="stable_cascade_super_resolution.safetensors"),
    HuggingFile("limingcv/ControlNet-Plus-Plus", "checkpoints/canny/controlnet/diffusion_pytorch_model.safetensors", save_with_filename="ControlNet-Plus-Plus_sd15_canny.safetensors", repo_type="space"),
    HuggingFile("limingcv/ControlNet-Plus-Plus", "checkpoints/depth/controlnet/diffusion_pytorch_model.safetensors", save_with_filename="ControlNet-Plus-Plus_sd15_grayscale_depth.safetensors", repo_type="space"),
    HuggingFile("limingcv/ControlNet-Plus-Plus", "checkpoints/hed/controlnet/diffusion_pytorch_model.bin", save_with_filename="ControlNet-Plus-Plus_sd15_hed.bin", repo_type="space"),
    HuggingFile("limingcv/ControlNet-Plus-Plus", "checkpoints/lineart/controlnet/diffusion_pytorch_model.bin", save_with_filename="ControlNet-Plus-Plus_sd15_lineart.bin", repo_type="space"),
    HuggingFile("limingcv/ControlNet-Plus-Plus", "checkpoints/seg/controlnet/diffusion_pytorch_model.safetensors", save_with_filename="ControlNet-Plus-Plus_sd15_ade20k_seg.safetensors", repo_type="space"),
    HuggingFile("xinsir/controlnet-scribble-sdxl-1.0", "diffusion_pytorch_model.safetensors", save_with_filename="xinsir-controlnet-scribble-sdxl-1.0.safetensors"),
    HuggingFile("xinsir/controlnet-canny-sdxl-1.0", "diffusion_pytorch_model.safetensors", save_with_filename="xinsir-controlnet-canny-sdxl-1.0.safetensors"),
    HuggingFile("xinsir/controlnet-canny-sdxl-1.0", "diffusion_pytorch_model_V2.safetensors", save_with_filename="xinsir-controlnet-canny-sdxl-1.0_V2.safetensors"),
    HuggingFile("xinsir/controlnet-openpose-sdxl-1.0", "diffusion_pytorch_model.safetensors", save_with_filename="xinsir-controlnet-openpose-sdxl-1.0.safetensors"),
    HuggingFile("xinsir/anime-painter", "diffusion_pytorch_model.safetensors", save_with_filename="xinsir-anime-painter-scribble-sdxl-1.0.safetensors"),
    HuggingFile("TheMistoAI/MistoLine", "mistoLine_rank256.safetensors"),
    HuggingFile("xinsir/controlnet-union-sdxl-1.0", "diffusion_pytorch_model_promax.safetensors", save_with_filename="xinsir-controlnet-union-sdxl-1.0-promax.safetensors"),
    HuggingFile("xinsir/controlnet-union-sdxl-1.0", "diffusion_pytorch_model.safetensors", save_with_filename="xinsir-controlnet-union-sdxl-1.0.safetensors"),
    HuggingFile("InstantX/FLUX.1-dev-Controlnet-Canny", "diffusion_pytorch_model.safetensors", save_with_filename="instantx-flux.1-dev-controlnet-canny.safetensors"),
    HuggingFile("InstantX/FLUX.1-dev-Controlnet-Union", "diffusion_pytorch_model.safetensors", save_with_filename="instantx-flux.1-dev-controlnet-union.safetensors"),
    HuggingFile("Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro", "diffusion_pytorch_model.safetensors", save_with_filename="shakker-labs-flux.1-dev-controlnet-union-pro.safetensors"),
    HuggingFile("TheMistoAI/MistoLine_Flux.dev", "mistoline_flux.dev_v1.safetensors"),
    HuggingFile("XLabs-AI/flux-controlnet-collections", "flux-canny-controlnet-v3.safetensors"),
    HuggingFile("XLabs-AI/flux-controlnet-collections", "flux-depth-controlnet-v3.safetensors"),
    HuggingFile("XLabs-AI/flux-controlnet-collections", "flux-hed-controlnet-v3.safetensors"),
    HuggingFile("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha", "diffusion_pytorch_model.safetensors", save_with_filename="alimama-creative-flux.1-dev-controlnet-inpainting-alpha.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-3.5-controlnets", "sd3.5_large_controlnet_canny.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-3.5-controlnets", "sd3.5_large_controlnet_depth.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-3.5-controlnets", "sd3.5_large_controlnet_blur.safetensors"),
    HuggingFile("Shakker-Labs/FLUX.1-dev-ControlNet-Depth", "diffusion_pytorch_model.safetensors", save_with_filename="shakker-labs-flux.1-dev-controlnet-depth.safetensors"),
    # WanVideo 2.2 controlnets
    HuggingFile("TheDenk/wan2.2-ti2v-5b-controlnet-depth-v1", "diffusion_pytorch_model.safetensors", save_with_filename="wan2.2-ti2v-5b-controlnet-depth-v1/diffusion_pytorch_model.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-InstantX-ControlNets", "split_files/controlnet/Qwen-Image-InstantX-ControlNet-Inpainting.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-InstantX-ControlNets", "split_files/controlnet/Qwen-Image-InstantX-ControlNet-Union.safetensors"),
    # --- Models referenced by comfyui-workflow-templates (controlnet) ---
    # Alibaba PAI "Fun" Controlnet-Union packs
    HuggingFile("alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union", "Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors"),
    HuggingFile("alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union", "Z-Image-Turbo-Fun-Controlnet-Union.safetensors"),
], folder_name="controlnet")

KNOWN_DIFF_CONTROLNETS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_canny_fp16.safetensors"),
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_depth_fp16.safetensors"),
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_hed_fp16.safetensors"),
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_mlsd_fp16.safetensors"),
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_normal_fp16.safetensors"),
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_openpose_fp16.safetensors"),
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_scribble_fp16.safetensors"),
    HuggingFile("kohya-ss/ControlNet-diff-modules", "diff_control_sd15_seg_fp16.safetensors"),
], folder_name="diff_controlnet")

KNOWN_APPROX_VAES: Final[KnownDownloadables] = KnownDownloadables([
    UrlFile("https://raw.githubusercontent.com/madebyollin/taesd/main/taeqi2_1_encoder.pth", show_in_ui=False),
    UrlFile("https://raw.githubusercontent.com/madebyollin/taesd/main/taeqi2_1_decoder.pth", show_in_ui=False),
    HuggingFile("madebyollin/taesd", "taesd_decoder.safetensors", show_in_ui=False),
    HuggingFile("madebyollin/taesd", "taesd_encoder.safetensors", show_in_ui=False),
    HuggingFile("madebyollin/taesdxl", "taesdxl_decoder.safetensors", show_in_ui=False),
    HuggingFile("madebyollin/taesdxl", "taesdxl_encoder.safetensors", show_in_ui=False),
    # todo: these are both the encoder and decoder, so it is not clear what should be done here
    # HuggingFile("madebyollin/taef1", "diffusion_pytorch_model.safetensors", save_with_filename="taef1_decoder.safetensors", show_in_ui=False),
    # HuggingFile("madebyollin/taesd3", "diffusion_pytorch_model.safetensors", save_with_filename="taesd3_decoder.safetensors", show_in_ui=False),
    UrlFile("https://raw.githubusercontent.com/madebyollin/taesd/main/taesd_decoder.pth", show_in_ui=False),
    UrlFile("https://raw.githubusercontent.com/madebyollin/taesd/main/taesd_encoder.pth", show_in_ui=False),
    UrlFile("https://raw.githubusercontent.com/madebyollin/taesd/main/taesdxl_encoder.pth", show_in_ui=False),
    UrlFile("https://raw.githubusercontent.com/madebyollin/taesd/main/taesdxl_decoder.pth", show_in_ui=False),
    UrlFile("https://raw.githubusercontent.com/madebyollin/taesd/main/taesd3_encoder.pth", show_in_ui=False),
    UrlFile("https://raw.githubusercontent.com/madebyollin/taesd/main/taesd3_decoder.pth", show_in_ui=False),
    UrlFile("https://raw.githubusercontent.com/madebyollin/taesd/main/taef1_encoder.pth", show_in_ui=False),
    UrlFile("https://raw.githubusercontent.com/madebyollin/taesd/main/taef1_decoder.pth", show_in_ui=False),
    # todo: update this with the video VAEs
    # WanVideoWrapper (Kijai) -- tiny VAE
    HuggingFile("Kijai/WanVideo_comfy", "taew2_1.safetensors", show_in_ui=False),
], folder_name="vae_approx")

KNOWN_VAES: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Comfy-Org/MOSS-Image", "vae/flux2-vae_small.safetensors"),
    HuggingFile("Comfy-Org/Ming-Image", "vae/ming_image_vae_bf16.safetensors"),
    HuggingFile("Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-2.1", "vae/qwen_image_2.1_vae_bf16.safetensors"),
    HuggingFile("Comfy-Org/marigold-v2-0", "vae/marigold_v2_albedo_vae.safetensors"),
    HuggingFile("Comfy-Org/marigold-v2-0", "vae/marigold_v2_depth_log_stage2_vae.safetensors"),
    HuggingFile("Comfy-Org/marigold-v2-0", "vae/marigold_v2_normals_vae.safetensors"),
    HuggingFile("Comfy-Org/MiniMax-Music-3", "vae/minimax_music3_dav.safetensors"),
    HuggingFile("Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_fp16.safetensors"),
    HuggingFile("Comfy-Org/MiniMax-H3", "vae/minimax_h3_audio_vae_fp32.safetensors"),
    HuggingFile("Comfy-Org/Mage-Flow", "vae/mage_flow_vae_bf16.safetensors"),
    HuggingFile("stabilityai/sdxl-vae", "sdxl_vae.safetensors", alternate_filenames=("sdxl.fp16.vae.safetensors",)),
    HuggingFile("stabilityai/sd-vae-ft-mse-original", "vae-ft-mse-840000-ema-pruned.safetensors"),
    # this is the flux VAE
    HuggingFile("Comfy-Org/HiDream-I1_ComfyUI", "split_files/vae/ae.safetensors"),
    HuggingFile("Comfy-Org/mochi_preview_repackaged", "split_files/vae/mochi_vae.safetensors"),
    HuggingFile("Comfy-Org/HunyuanVideo_repackaged", "split_files/vae/hunyuan_video_vae_bf16.safetensors", alternate_filenames=("hyvid/hunyuan_video_vae_bf16.safetensors",)),
    HuggingFile("comfyanonymous/cosmos_1.0_text_encoder_and_VAE_ComfyUI", "vae/cosmos_cv8x8x8_1.0.safetensors"),
    HuggingFile("Comfy-Org/Lumina_Image_2.0_Repackaged", "split_files/vae/ae.safetensors", save_with_filename="lumina_image_2.0-ae.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/vae/wan_2.1_vae.safetensors", alternate_filenames=("vae/wan_2.1_vae.safetensors",)),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/vae/wan2.2_vae.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors", alternate_filenames=("vae/qwen_image_vae.safetensors",)),
    HuggingFile("Comfy-Org/Boogu-Image", "vae/flux1_vae_bf16.safetensors"),
    HuggingFile("Comfy-Org/ace_step_1.5_ComfyUI_files", "split_files/vae/ace_1.5_vae.safetensors"),
    # Flux 2
    HuggingFile("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors"),
    HuggingFile("black-forest-labs/FLUX.2-small-decoder", "full_encoder_small_decoder.safetensors"),
    HuggingFile("VAST-AI/TripoSplat", "vae/flux2-vae.safetensors"),
    HuggingFile("VAST-AI/TripoSplat", "vae/triposplat_vae_decoder_fp16.safetensors"),
    HuggingFile("Comfy-Org/Pixal3D", "vae/trellis_2_texture_vae_bf16.safetensors"),
    HuggingFile("Comfy-Org/Pixal3D", "vae/trellis_2_shape_vae_bf16.safetensors"),
    # LTX-2.5
    HuggingFile("Lightricks/LTX-2.5", "vae/ltx-2.5-video-vae-bf16.safetensors"),
    HuggingFile("Lightricks/LTX-2.5", "vae/ltx-2.5-audio-vae-bf16.safetensors"),
    # Z Image Turbo
    HuggingFile("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors", save_with_filename="z_image_turbo_vae.safetensors"),
    # SeedVR2
    HuggingFile(
        "Comfy-Org/SeedVR2",
        "vae/seedvr2_ema_vae_fp16.safetensors",
        alternate_filenames=("ema_vae_fp16.safetensors",),
    ),
    # Void video inpainting
    HuggingFile("Comfy-Org/void-model", "vae/cogvideox_vae.safetensors"),
    # Hunyuan Image
    HuggingFile("Comfy-Org/HunyuanImage_2.1_ComfyUI", "split_files/vae/hunyuan_image_2.1_vae_fp16.safetensors"),
    HuggingFile("Comfy-Org/HunyuanImage_2.1_ComfyUI", "split_files/vae/hunyuan_image_refiner_vae_fp16.safetensors"),
    HuggingFile("PedroMarinhoDev/HunyuanImage-3.0-Instruct-ComfyUI", "vae/hunyuan_image_3_vae_fp16.safetensors"),
    HuggingFile("Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/vae/hunyuanvideo15_vae_fp16.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-Layered_ComfyUI", "split_files/vae/qwen_image_layered_vae.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/vae/wan_alpha_2.1_vae_alpha_channel.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/vae/wan_alpha_2.1_vae_rgb_channel.safetensors"),
    # WanVideoWrapper (Kijai) -- VAE
    HuggingFile("Kijai/WanVideo_comfy", "Wan2_1_VAE_bf16.safetensors", alternate_filenames=("wanvideo/Wan2_1_VAE_bf16.safetensors",)),
    HuggingFile("Kijai/WanVideo_comfy", "Wan2_2_VAE_bf16.safetensors", alternate_filenames=("wanvideo/Wan2_2_VAE_bf16.safetensors",)),
    HuggingFile("Kijai/WanVideo_comfy", "FlashVSR/Wan2_1_FlashVSR_TCDecoder_fp32.safetensors", save_with_filename="Wan2_1_FlashVSR_TCDecoder_fp32.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "MTVCrafter/WanVideo_MTV_Crafter_4DMoT_VQVAE_fp32.safetensors", save_with_filename="wanvideo/WanVideo_MTV_Crafter_4DMoT_VQVAE_fp32.safetensors", alternate_filenames=("wanvideo/MTV_Crafter_4DMoT_VQVAE_fp32.safetensors",), show_in_ui=False),
], folder_name="vae")

KNOWN_HUGGINGFACE_MODEL_REPOS: Final[Set[str]] = {
    'JingyeChen22/textdiffuser2_layout_planner',
    'JingyeChen22/textdiffuser2-full-ft',
    'microsoft/Phi-4-mini-instruct',
    'llava-hf/llava-v1.6-mistral-7b-hf',
    'facebook/nllb-200-distilled-1.3B',
    'THUDM/chatglm3-6b',
    'roborovski/superprompt-v1',
    'Qwen/Qwen2-VL-7B-Instruct',
    'microsoft/Florence-2-large-ft',
    'google/paligemma2-10b-pt-896',
    'google/paligemma2-28b-pt-896',
    'google/paligemma-3b-ft-refcoco-seg-896',
    'microsoft/phi-4',
    'appmana/Cosmos-1.0-Prompt-Upsampler-12B-Text2World-hf',
    'llava-hf/llava-onevision-qwen2-7b-si-hf',
    'llava-hf/llama3-llava-next-8b-hf',
    'PromptEnhancer/PromptEnhancer-32B',
    # HiDream-O1 publishes complete sharded Transformers repos.
    'HiDream-ai/HiDream-O1-Image',
    'HiDream-ai/HiDream-O1-Image-Dev',
    # Florence-2 (ComfyUI-Florence2)
    'microsoft/Florence-2-base',
    'microsoft/Florence-2-base-ft',
    'microsoft/Florence-2-large',
    'thwri/CogFlorence-2.1-Large',
    'thwri/CogFlorence-2.2-Large',
    'gokaygokay/Florence-2-SD3-Captioner',
    'gokaygokay/Florence-2-Flux-Large',
    'MiaoshouAI/Florence-2-base-PromptGen-v1.5',
    'MiaoshouAI/Florence-2-base-PromptGen-v2.0',
    'MiaoshouAI/Florence-2-large-PromptGen-v1.5',
    'MiaoshouAI/Florence-2-large-PromptGen-v2.0',
    # NormalCrafter (ComfyUI-NormalCrafterWrapper)
    'Yanrui95/NormalCrafter',
    # ChatterBox TTS (ComfyUI_Fill-ChatterBox)
    'ResembleAI/chatterbox',
    'ResembleAI/chatterbox-turbo',
    # Ideogram 4 publishes complete Diffusers repos.
    'ideogram-ai/ideogram-4-nf4',
    'ideogram-ai/ideogram-4-fp8',
}

KNOWN_UNET_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Comfy-Org/MOSS-Image", "diffusion_models/moss_image_base_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/MOSS-Image", "diffusion_models/moss_image_sft_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/Ming-Image", "diffusion_models/ming_image_0.1_design_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-2.1", "diffusion_models/qwen_image_2.1_bf16.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-2.1", "diffusion_models/qwen_image_2.1_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-2.1", "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/marigold-v2-0", "diffusion_models/qwen_image_edit_2509_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/MiniMax-Music-3", "diffusion_models/minimax_music3_dit_fp16.safetensors"),
    HuggingFile("Comfy-Org/MiniMax-Music-3", "diffusion_models/minimax_music3_dit_fp32.safetensors"),
    HuggingFile("Comfy-Org/MiniMax-Music-3", "diffusion_models/minimax_music3_dit_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/MiniMax-H3", "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/MiniMax-H3", "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/Mage-Flow", "diffusion_models/mage_flow_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/Mage-Flow", "diffusion_models/mage_flow_turbo_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/Mage-Flow", "diffusion_models/mage_flow_edit_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/Mage-Flow", "diffusion_models/mage_flow_edit_turbo_int8_convrot.safetensors"),
    HuggingFile("ByteDance/Hyper-SD", "Hyper-SDXL-1step-Unet-Comfyui.fp16.safetensors"),
    HuggingFile("black-forest-labs/FLUX.1-schnell", "flux1-schnell.safetensors", alternate_filenames=("main/flux1-schnell.safetensors",)),
    HuggingFile("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors", alternate_filenames=("main/flux1-dev.safetensors",)),
    HuggingFile("black-forest-labs/FLUX.1-Fill-dev", "flux1-fill-dev.safetensors", alternate_filenames=("main/flux1-fill-dev.safetensors",)),
    HuggingFile("black-forest-labs/FLUX.1-Canny-dev", "flux1-canny-dev.safetensors", alternate_filenames=("main/flux1-canny-dev.safetensors",)),
    HuggingFile("black-forest-labs/FLUX.1-Depth-dev", "flux1-depth-dev.safetensors", alternate_filenames=("main/flux1-depth-dev.safetensors",)),
    HuggingFile("black-forest-labs/FLUX.1-Kontext-dev", "flux1-kontext-dev.safetensors"),
    HuggingFile("Kijai/flux-fp8", "flux1-dev-fp8.safetensors"),
    HuggingFile("Kijai/flux-fp8", "flux1-schnell-fp8.safetensors"),
    HuggingFile("Comfy-Org/mochi_preview_repackaged", "split_files/diffusion_models/mochi_preview_bf16.safetensors"),
    HuggingFile("Comfy-Org/mochi_preview_repackaged", "split_files/diffusion_models/mochi_preview_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/HunyuanVideo_repackaged", "split_files/diffusion_models/hunyuan_video_t2v_720p_bf16.safetensors", alternate_filenames=("hyvideo/hunyuan_video_720_bf16.safetensors",)),
    HuggingFile("Kijai/HunyuanVideo_comfy", "hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors", save_with_filename="hunyuan_video_720_fp8_e4m3fn.safetensors", alternate_filenames=("hyvideo/hunyuan_video_720_fp8_e4m3fn.safetensors",), show_in_ui=False),
    HuggingFile("Comfy-Org/HunyuanVideo_repackaged", "split_files/diffusion_models/hunyuan_video_image_to_video_720p_bf16.safetensors"),
    HuggingFile("mcmonkey/cosmos-1.0", "Cosmos-1_0-Diffusion-14B-Text2World.safetensors"),
    HuggingFile("mcmonkey/cosmos-1.0", "Cosmos-1_0-Diffusion-14B-Video2World.safetensors"),
    HuggingFile("mcmonkey/cosmos-1.0", "Cosmos-1_0-Diffusion-7B-Text2World.safetensors"),
    HuggingFile("mcmonkey/cosmos-1.0", "Cosmos-1_0-Diffusion-7B-Video2World.safetensors"),
    HuggingFile("Comfy-Org/Lumina_Image_2.0_Repackaged", "split_files/diffusion_models/lumina_2_model_bf16.safetensors"),
    HuggingFile("Comfy-Org/Krea-2", "diffusion_models/krea2_raw_bf16.safetensors"),
    HuggingFile("Comfy-Org/Krea-2", "diffusion_models/krea2_raw_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Krea-2", "diffusion_models/krea2_turbo_bf16.safetensors"),
    HuggingFile("Comfy-Org/Krea-2", "diffusion_models/krea2_turbo_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Krea-2", "diffusion_models/krea2_turbo_int8_convrot.safetensors"),
    # LTX-2.5
    HuggingFile("Lightricks/LTX-2.5", "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"),
    # SeedVR2
    HuggingFile("Comfy-Org/SeedVR2", "diffusion_models/seedvr2_3b_fp16.safetensors"),
    HuggingFile("Comfy-Org/SeedVR2", "diffusion_models/seedvr2_3b_fp8_e4m3fn.safetensors"),
    HuggingFile("Comfy-Org/SeedVR2", "diffusion_models/seedvr2_3b_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/SeedVR2", "diffusion_models/seedvr2_3b_mxfp8.safetensors"),
    HuggingFile("Comfy-Org/SeedVR2", "diffusion_models/seedvr2_3b_nvfp4.safetensors"),
    HuggingFile("Comfy-Org/SeedVR2", "diffusion_models/seedvr2_7b_fp16.safetensors"),
    HuggingFile("Comfy-Org/SeedVR2", "diffusion_models/seedvr2_7b_fp8_e4m3fn.safetensors"),
    HuggingFile("Comfy-Org/SeedVR2", "diffusion_models/seedvr2_7b_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/SeedVR2", "diffusion_models/seedvr2_7b_mxfp8.safetensors"),
    HuggingFile("Comfy-Org/SeedVR2", "diffusion_models/seedvr2_7b_nvfp4.safetensors"),
    HuggingFile("Comfy-Org/SeedVR2", "diffusion_models/seedvr2_7b_sharp_fp16.safetensors"),
    HuggingFile("Comfy-Org/SeedVR2", "diffusion_models/seedvr2_7b_sharp_fp8_e4m3fn.safetensors"),
    HuggingFile("Comfy-Org/SeedVR2", "diffusion_models/seedvr2_7b_sharp_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/SeedVR2", "diffusion_models/seedvr2_7b_sharp_mxfp8.safetensors"),
    HuggingFile("Comfy-Org/SeedVR2", "diffusion_models/seedvr2_7b_sharp_nvfp4.safetensors"),
    HuggingFile("Winnougan/Krea-2-Base-Turbo-NVFP4-FP8-INT8", "Krea2_Raw_convrot_int8mixed.safetensors"),
    HuggingFile("Winnougan/Krea-2-Base-Turbo-NVFP4-FP8-INT8", "Krea2_Turbo_convrot_int8mixed.safetensors"),
    HuggingFile("Comfy-Org/Boogu-Image", "diffusion_models/boogu_image_base_bf16.safetensors"),
    HuggingFile("Comfy-Org/Boogu-Image", "diffusion_models/boogu_image_base_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Boogu-Image", "diffusion_models/boogu_image_edit_bf16.safetensors"),
    HuggingFile("Comfy-Org/Boogu-Image", "diffusion_models/boogu_image_edit_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Boogu-Image", "diffusion_models/boogu_image_edit_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/Boogu-Image", "diffusion_models/boogu_image_turbo_bf16.safetensors"),
    HuggingFile("Comfy-Org/Boogu-Image", "diffusion_models/boogu_image_turbo_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors", show_in_ui=False),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_i2v_720p_14B_bf16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_i2v_720p_14B_fp16.safetensors", show_in_ui=False),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_i2v_720p_14B_fp8_e4m3fn.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors", show_in_ui=False),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_t2v_14B_bf16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_t2v_14B_fp16.safetensors", show_in_ui=False),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_t2v_14B_fp8_e4m3fn.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_t2v_14B_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/SCAIL-2", "diffusion_models/wan2.1_14B_SCAIL_2_fp16.safetensors", show_in_ui=False),
    HuggingFile("Comfy-Org/SCAIL-2", "diffusion_models/wan2.1_14B_SCAIL_2_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/SCAIL-2", "diffusion_models/wan2.1_14B_SCAIL_2_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/Wan-Dancer", "diffusion_models/wan2.2_dancer_14b_global_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Wan-Dancer", "diffusion_models/wan2.2_dancer_14b_local_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/HunyuanVideo_repackaged", "split_files/diffusion_models/hunyuan_video_v2_replace_image_to_video_720p_bf16.safetensors"),
    HuggingFile("Comfy-Org/HiDream-I1_ComfyUI", "split_files/diffusion_models/hidream_i1_dev_bf16.safetensors"),
    HuggingFile("Comfy-Org/HiDream-I1_ComfyUI", "split_files/diffusion_models/hidream_i1_dev_fp8.safetensors"),
    HuggingFile("Comfy-Org/HiDream-I1_ComfyUI", "split_files/diffusion_models/hidream_i1_fast_fp8.safetensors"),
    HuggingFile("Comfy-Org/HiDream-I1_ComfyUI", "split_files/diffusion_models/hidream_i1_full_fp16.safetensors"),
    HuggingFile("Comfy-Org/HiDream-I1_ComfyUI", "split_files/diffusion_models/hidream_i1_full_fp8.safetensors"),
    HuggingFile("Comfy-Org/HiDream-I1_ComfyUI", "split_files/diffusion_models/hidream_e1_full_bf16.safetensors"),
    HuggingFile("Comfy-Org/HiDream-I1_ComfyUI", "split_files/diffusion_models/hidream_e1_1_bf16.safetensors"),
    HuggingFile("Comfy-Org/Cosmos_Predict2_repackaged", "cosmos_predict2_2B_t2i.safetensors"),
    HuggingFile("Comfy-Org/Cosmos_Predict2_repackaged", "cosmos_predict2_14B_t2i.safetensors"),
    HuggingFile("Comfy-Org/Cosmos_Predict2_repackaged", "cosmos_predict2_2B_video2world_480p_16fps.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_fun_camera_v1.1_1.3B_bf16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_fun_camera_v1.1_14B_bf16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_flf2v_720p_14B_fp16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_fun_control_1.3B_bf16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_fun_inp_1.3B_bf16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Bernini-R", "diffusion_models/wan2.2_bernini_r_high_noise_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Bernini-R", "diffusion_models/wan2.2_bernini_r_low_noise_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors", alternate_filenames=("WanVideo/2_2/wan2.2_ti2v_5B_fp16.safetensors",)),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_fun_camera_high_noise_14B_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_fun_camera_low_noise_14B_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_fun_control_5B_bf16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_fun_inpaint_5B_bf16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_fun_inpaint_high_noise_14B_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_fun_inpaint_low_noise_14B_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors"),
    HuggingFile("lodestones/Chroma", "chroma-unlocked-v37.safetensors"),
    HuggingFile("QuantStack/Wan2.2-T2V-A14B-GGUF", "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf"),
    HuggingFile("QuantStack/Wan2.2-T2V-A14B-GGUF", "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf"),
    HuggingFile("QuantStack/Wan2.2-T2V-A14B-GGUF", "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf"),
    HuggingFile("QuantStack/Wan2.2-T2V-A14B-GGUF", "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf"),
    HuggingFile("city96/Qwen-Image-gguf", "qwen-image-Q4_K_M.gguf"),
    HuggingFile("city96/Qwen-Image-gguf", "qwen-image-Q8_0.gguf"),
    HuggingFile("Comfy-Org/Qwen-Image_ComfyUI", "split_files/diffusion_models/qwen_image_bf16.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image_ComfyUI", "split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image_ComfyUI", "split_files/diffusion_models/qwen_image_2512_bf16.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image_ComfyUI", "split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image_ComfyUI", "non_official/diffusion_models/qwen_image_distill_full_bf16.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image_ComfyUI", "non_official/diffusion_models/qwen_image_distill_full_fp8_e4m3fn.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/diffusion_models/qwen_image_edit_2509_bf16.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/diffusion_models/qwen_image_edit_bf16.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors"),
    # Flux 2
    HuggingFile("Comfy-Org/flux2-dev", "split_files/diffusion_models/flux2_dev_fp8mixed.safetensors"),
    # Uploaded to the official repository in discussion/PR 14. Keep the
    # immutable revision until Comfy-Org merges the artifact into main.
    HuggingFile(
        "Comfy-Org/flux2-dev",
        "split_files/diffusion_models/flux2-dev-int8-convrot-simple.safetensors",
        revision="14cd543dbc65aa53b21220e0c67c58d6c64bbdb9",
        size=33055836040,
    ),
    HuggingFile("black-forest-labs/FLUX.2-klein-base-4B", "flux-2-klein-base-4b.safetensors"),
    HuggingFile("Comfy-Org/flux2-klein", "split_files/diffusion_models/flux-2-klein-4b.safetensors"),
    HuggingFile("black-forest-labs/flux-3-image-klein-comfy", "diffusion_models/flux3_image_klein_int8_convrot.safetensors", revision="849a7a417c54120ac12f249da656b84ec05fd7eb", size=5807427339),
    HuggingFile("black-forest-labs/flux-3-image-klein-comfy", "diffusion_models/flux3_image_klein_sd_int8_convrot.safetensors", revision="849a7a417c54120ac12f249da656b84ec05fd7eb", size=5827874779),
    HuggingFile("black-forest-labs/flux-3-image-klein-comfy", "diffusion_models/flux3_image_klein_base_int8_convrot.safetensors", revision="5c9b714851dc072b04a23ef6d2d36421f5ccb3a0", size=5807427339),
    HuggingFile("black-forest-labs/flux-3-image-klein-comfy", "diffusion_models/flux3_image_klein_base_bf16.safetensors", revision="5c9b714851dc072b04a23ef6d2d36421f5ccb3a0", size=11076957072),
    HuggingFile("black-forest-labs/flux-3-image-klein-comfy", "diffusion_models/flux3_image_klein_bf16.safetensors", revision="5c9b714851dc072b04a23ef6d2d36421f5ccb3a0", size=11076957072),
    HuggingFile("black-forest-labs/flux-3-image-klein-comfy", "diffusion_models/flux3_image_klein_sd_bf16.safetensors", revision="5c9b714851dc072b04a23ef6d2d36421f5ccb3a0", size=11097404512),
    HuggingFile("Comfy-Org/MOSS-Image", "diffusion_models/moss_image_base_bf16.safetensors", revision="9df12d99a97c9818afc555125829f9ed40b87b24", size=10164655808),
    HuggingFile("Comfy-Org/MOSS-Image", "diffusion_models/moss_image_sft_bf16.safetensors", revision="9df12d99a97c9818afc555125829f9ed40b87b24", size=10164655808),
    HuggingFile("Comfy-Org/MOSS-Image", "diffusion_models/moss_image_base_int8_convrot_attn.safetensors", revision="9df12d99a97c9818afc555125829f9ed40b87b24", size=5413698958),
    HuggingFile("Comfy-Org/MOSS-Image", "diffusion_models/moss_image_sft_int8_convrot_attn.safetensors", revision="9df12d99a97c9818afc555125829f9ed40b87b24", size=5413698958),
    # briaai/Fibo-2.0-dev 5306044 transformer, its two diffusers shards as one file (gated, non-commercial licence: a private repository)
    HuggingFile("appmana/Fibo-2.0-ComfyUI", "diffusion_models/fibo2_dev_bf16.safetensors", revision="e51ce9667827b30d3c31290f57b81a0626910596", size=16254770194),
    HuggingFile("Comfy-Org/Fibo-2.0", "diffusion_models/fibo2_dev_int8_convrot.safetensors", revision="b2bffaf57d48825a6989b986f24d5cfb8c71231b", size=8408562474),
    HuggingFile("Comfy-Org/Fibo-2.0", "diffusion_models/fibo2_edit_turbo_int8_convrot.safetensors", revision="b2bffaf57d48825a6989b986f24d5cfb8c71231b", size=8408562474),
    HuggingFile("black-forest-labs/FLUX.2-klein-4b-fp8", "flux-2-klein-4b-fp8.safetensors"),
    HuggingFile("black-forest-labs/FLUX.2-klein-9b-fp8", "flux-2-klein-9b-fp8.safetensors"),
    HuggingFile("black-forest-labs/FLUX.2-klein-9b-kv-fp8", "flux-2-klein-9b-kv-fp8.safetensors"),
    HuggingFile("black-forest-labs/FLUX.2-klein-base-4b-fp8", "flux-2-klein-base-4b-fp8.safetensors"),
    HuggingFile("black-forest-labs/FLUX.2-klein-base-9b-fp8", "flux-2-klein-base-9b-fp8.safetensors"),
    # TripoSplat
    HuggingFile("VAST-AI/TripoSplat", "diffusion_models/triposplat_fp16.safetensors"),
    # ERNIE Image (Comfy-Org)
    HuggingFile("Comfy-Org/ERNIE-Image", "diffusion_models/ernie-image-turbo.safetensors"),
    # Z Image Turbo
    HuggingFile("Comfy-Org/z_image_turbo", "split_files/diffusion_models/z_image_turbo_bf16.safetensors"),
    HuggingFile("Comfy-Org/z_image_turbo", "split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/z_image", "split_files/diffusion_models/z_image_bf16.safetensors"),
    HuggingFile("Comfy-Org/z_image", "split_files/diffusion_models/z_image_int8_convrot.safetensors"),
    # Omnigen 2
    HuggingFile("Comfy-Org/Omnigen2_ComfyUI_repackaged", "split_files/diffusion_models/omnigen2_fp16.safetensors"),
    # Hunyuan Image
    HuggingFile("Comfy-Org/HunyuanImage_2.1_ComfyUI", "split_files/diffusion_models/hunyuanimage2.1_bf16.safetensors"),
    HuggingFile("Comfy-Org/HunyuanImage_2.1_ComfyUI", "split_files/diffusion_models/hunyuanimage2.1_refiner_bf16.safetensors"),
    # HunyuanImage 3.0 Instruct, for the comfyui-hunyuanimage3 node
    HuggingFile("PedroMarinhoDev/HunyuanImage-3.0-Instruct-ComfyUI", "hunyuan_image_3_instruct_bf16.safetensors"),
    HuggingFile("PedroMarinhoDev/HunyuanImage-3.0-Instruct-ComfyUI", "hunyuan_image_3_instruct_w4a8.safetensors"),
    HuggingFile("Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/diffusion_models/capybara_v0.1.safetensors"),
    HuggingFile("Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/diffusion_models/hunyuanvideo1.5_1080p_sr_distilled_fp16.safetensors"),
    HuggingFile("Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/diffusion_models/hunyuanvideo1.5_720p_i2v_fp16.safetensors"),
    HuggingFile("Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/diffusion_models/hunyuanvideo1.5_720p_t2v_fp16.safetensors"),
    HuggingFile("Comfy-Org/LongCat-Image", "split_files/diffusion_models/longcat_image_bf16.safetensors"),
    HuggingFile("Comfy-Org/ace_step_1.5_ComfyUI_files", "split_files/diffusion_models/acestep_v1.5_turbo.safetensors"),
    HuggingFile("Comfy-Org/ace_step_1.5_ComfyUI_files", "split_files/diffusion_models/acestep_v1.5_xl_base_bf16.safetensors"),
    HuggingFile("Comfy-Org/ace_step_1.5_ComfyUI_files", "split_files/diffusion_models/acestep_v1.5_xl_sft_bf16.safetensors"),
    HuggingFile("Comfy-Org/ace_step_1.5_ComfyUI_files", "split_files/diffusion_models/acestep_v1.5_xl_turbo_bf16.safetensors"),
    HuggingFile("Comfy-Org/Chroma1-HD_repackaged", "split_files/diffusion_models/Chroma1-HD-fp8mixed.safetensors"),
    HuggingFile("Comfy-Org/Chroma1-Radiance_Repackaged", "split_files/diffusion_models/chroma-radiance-x0.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/chrono_edit_14B_fp16.safetensors"),
    HuggingFile("Comfy-Org/OneReward_repackaged", "split_files/diffusion_models/flux.1-fill-dev-OneReward-transformer_fp8.safetensors"),
    HuggingFile("Comfy-Org/flux1-kontext-dev_ComfyUI", "split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/FLUX.1-Krea-dev_ComfyUI", "split_files/diffusion_models/flux1-krea-dev_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/NewBie-image-Exp0.1_repackaged", "split_files/diffusion_models/NewBie-Image-Exp0.1-bf16.safetensors"),
    HuggingFile("Comfy-Org/JoyAI-Image-Edit", "diffusion_models/joyai_image_edit_bf16.safetensors"),
    HuggingFile("Comfy-Org/JoyAI-Image-Edit", "diffusion_models/joyai_image_edit_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-Layered_ComfyUI", "split_files/diffusion_models/qwen_image_layered_bf16.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-Layered_ComfyUI", "split_files/diffusion_models/qwen_image_layered_fp8mixed.safetensors"),
    HuggingFile("appmana/qwen-image-layered-int8convrot", "qwen_image_layered_int8convrot.safetensors"),
    HuggingFile("Comfy-Org/HuMo_ComfyUI", "split_files/diffusion_models/humo_17B_fp8_e4m3fn.safetensors"),
    HuggingFile("Kijai/LTXV2_comfy", "diffusion_models/ltx-2-19b-distilled_transformer_only_bf16.safetensors"),
    # Ovis
    HuggingFile("Comfy-Org/Ovis-Image", "split_files/diffusion_models/ovis_image_bf16.safetensors"),
    # WanVideoWrapper (Kijai) -- base fp8 models
    HuggingFile("Kijai/WanVideo_comfy", "Wan2_1-T2V-14B_fp8_e4m3fn.safetensors", save_with_filename="WanVideo/Wan2_1-T2V-14B_fp8_e4m3fn.safetensors"),
    HuggingFile("Kijai/WanVideo_comfy", "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors", save_with_filename="WanVideo/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"),
    HuggingFile("Kijai/WanVideo_comfy", "Wan2_1-I2V-14B-720P_fp8_e4m3fn.safetensors", save_with_filename="WanVideo/Wan2_1-I2V-14B-720P_fp8_e4m3fn.safetensors"),
    HuggingFile("Kijai/WanVideo_comfy", "Wan2_1-FLF2V-14B-720P_fp8_e4m3fn.safetensors", save_with_filename="WanVideo/Wan2_1-FLF2V-14B-720P_fp8_e4m3fn.safetensors"),
    HuggingFile("Kijai/WanVideo_comfy", "Wan2_1-VACE_module_1_3B_bf16.safetensors", save_with_filename="WanVideo/Wan2_1-VACE_module_1_3B_bf16.safetensors"),
    HuggingFile("Kijai/WanVideo_comfy", "Wan2_1-VACE_module_14B_bf16.safetensors", save_with_filename="WanVideo/Wan2_1-VACE_module_14B_bf16.safetensors", alternate_filenames=("Wan2_1-VACE_module_14B_bf16.safetensors",), show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "wan2.1_t2v_1.3B_fp16.safetensors", save_with_filename="WanVideo/wan2.1_t2v_1.3B_fp16.safetensors"),
    HuggingFile("Kijai/WanVideo_comfy", "wan2.1_fun_control_1.3B_bf16.safetensors", save_with_filename="WanVideo/wan2.1_fun_control_1.3B_bf16.safetensors", alternate_filenames=("wan2.1_fun_control_1.3B_bf16.safetensors",)),
    HuggingFile("Kijai/WanVideo_comfy", "Fun/Wan2.1-Fun-V1.1-1.3B-Control-Camera.safetensors", save_with_filename="WanVideo/Wan2.1-Fun-V1.1-1.3B-Control-Camera.safetensors"),
    HuggingFile("Kijai/WanVideo_comfy", "Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors", save_with_filename="WanVideo/Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors", alternate_filenames=("WanVideo/Wan2_1-Wan-I2V-ATI-14B_fp8_e4m3fn.safetensors", "Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors"), show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "SCAIL/Wan21-14B-SCAIL-preview_comfy_bf16.safetensors", save_with_filename="WanVideo/SCAIL/Wan21-14B-SCAIL-preview_comfy_bf16.safetensors", alternate_filenames=("Wan21-14B-SCAIL-preview_comfy_bf16.safetensors",), show_in_ui=False),
    # WanVideoWrapper -- specialty models
    HuggingFile("Kijai/WanVideo_comfy", "EchoShot/Wan2_1-T2V-1-3B-EchoShot_fp16.safetensors", save_with_filename="WanVideo/EchoShot/Wan2_1-T2V-1-3B-EchoShot_fp16.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "FantasyPortrait/Wan2_1_FantasyPortrait_fp16.safetensors", save_with_filename="WanVideo/FantasyPortrait/Wan2_1_FantasyPortrait_fp16.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "fantasytalking_fp16.safetensors", save_with_filename="WanVideo/fantasytalking_fp16.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "FlashVSR/Wan2_1-T2V-1_3B_FlashVSR_fp32.safetensors", save_with_filename="WanVideo/FlashVSR/Wan2_1-T2V-1_3B_FlashVSR_fp32.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "FlashVSR/Wan2_1_FlashVSR_LQ_proj_model_bf16.safetensors", save_with_filename="WanVideo/FlashVSR/Wan2_1_FlashVSR_LQ_proj_model_bf16.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "MTVCrafter/Wan2_1_MTV-Crafter_motion_adapter_bf16.safetensors", save_with_filename="WanVideo/MTVCrafter/Wan2_1_MTV-Crafter_motion_adapter_bf16.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "Skyreels/Wan2_1-SkyReels-V2-DF-1_3B-540P_fp32.safetensors", save_with_filename="WanVideo/Skyreels/Wan2_1-SkyReels-V2-DF-1_3B-540P_fp32.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "Skyreels/Wan2_1_SkyreelsA2_fp8_e4m3fn.safetensors", save_with_filename="WanVideo/Wan2_1_SkyreelsA2_fp8_e4m3fn.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "UniLumos/Wan2_1_UniLumos_1_3B_bf16.safetensors", save_with_filename="WanVideo/UniLumos/Wan2_1_UniLumos_1_3B_bf16.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "Wan2_1_kwai_recammaster_1_3B_step20000_bf16.safetensors", save_with_filename="WanVideo/Wan2_1_kwai_recammaster_1_3B_step20000_bf16.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "Lynx/lynx_lite_resampler_fp32.safetensors", save_with_filename="WanVideo/lynx/lynx_lite_resampler_fp32.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "Lynx/Wan2_1-T2V-Lynx_full_ref_layers_fp16.safetensors", save_with_filename="WanVideo/lynx/Wan2_1-T2V-Lynx_full_ref_layers_fp16.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "Lynx/Wan2_1-T2V-Lynx_lite_ip_layers_fp16.safetensors", save_with_filename="WanVideo/lynx/Wan2_1-T2V-Lynx_lite_ip_layers_fp16.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "FastWan/Wan2_2-TI2V-5B-FastWanFullAttn_bf16.safetensors", save_with_filename="Wan2_2-TI2V-5B-FastWanFullAttn_bf16.safetensors", show_in_ui=False),
    # WanVideoWrapper -- fp8 scaled (Kijai quantized)
    HuggingFile("Kijai/WanVideo_comfy_fp8_scaled", "T2V/Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors", save_with_filename="WanVideo/fp8_scaled_kj/T2V/Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors"),
    HuggingFile("Kijai/WanVideo_comfy_fp8_scaled", "I2V/Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors", save_with_filename="WanVideo/fp8_scaled_kj/I2V/Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors", alternate_filenames=("Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors",), show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy_fp8_scaled", "I2V/Wan2_1-I2V-14B-720p_fp8_e4m3fn_scaled_KJ.safetensors", save_with_filename="WanVideo/fp8_scaled_kj/I2V/Wan2_1-I2V-14B-720p_fp8_e4m3fn_scaled_KJ.safetensors"),
    HuggingFile("Kijai/WanVideo_comfy_fp8_scaled", "I2V/Wan2_1-I2V-14B-MAGREF_fp8_e4m3fn_scaled_KJ.safetensors", save_with_filename="WanVideo/fp8_scaled_kj/I2V/Wan2_1-I2V-14B-MAGREF_fp8_e4m3fn_scaled_KJ.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy_fp8_scaled", "T2V/Wan2_1-T2V-14B-Phantom_fp8_e4m3fn_scaled_KJ.safetensors", save_with_filename="WanVideo/Wan2_1-T2V-14B-Phantom_fp8_e4m3fn_scaled_KJ.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy_fp8_scaled", "I2V/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors", save_with_filename="WanVideo/2_2/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors"),
    HuggingFile("Kijai/WanVideo_comfy_fp8_scaled", "I2V/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors", save_with_filename="WanVideo/2_2/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors"),
    HuggingFile("Kijai/WanVideo_comfy_fp8_scaled", "T2V/Wan2_2-T2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors", save_with_filename="WanVideo/2_2/Wan2_2-T2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors"),
    HuggingFile("Kijai/WanVideo_comfy_fp8_scaled", "T2V/Wan2_2-T2V-A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors", save_with_filename="WanVideo/2_2/Wan2_2-T2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors", alternate_filenames=("WanVideo/2_2/Wan2_2-T2V-A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors",)),
    HuggingFile("Kijai/WanVideo_comfy_fp8_scaled", "Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors", save_with_filename="WanVideo/2_2/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors", alternate_filenames=("Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors",), show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy_fp8_scaled", "Fun/Wan2_2-Fun-Control-A14B-HIGH_fp8_e4m3fn_scaled_KJ_fixed.safetensors", save_with_filename="WanVideo/2_2/Fun/Wan2_2-Fun-Control-A14B-HIGH_fp8_e4m3fn_scaled_KJ_fixed.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy_fp8_scaled", "Fun/Wan2_2-Fun-Control-A14B-LOW_fp8_e4m3fn_scaled_KJ_fixed.safetensors", save_with_filename="WanVideo/2_2/Fun/Wan2_2-Fun-Control-A14B-LOW_fp8_e4m3fn_scaled_KJ_fixed.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy_fp8_scaled", "Fun/Wan2_2-Fun-Control-Camera-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors", save_with_filename="WanVideo/2_2/Fun/Wan2_2-Fun-Control-Camera-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy_fp8_scaled", "HuMo/Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors", save_with_filename="WanVideo/HuMo/Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy_fp8_scaled", "SCAIL/Wan21-14B-SCAIL-preview_fp8_e4m3fn_scaled_KJ.safetensors", save_with_filename="WanVideo/SCAIL/Wan21-14B-SCAIL-preview_fp8_e4m3fn_scaled_KJ.safetensors", alternate_filenames=("Wan21-14B-SCAIL-preview_fp8_e4m3fn_scaled_KJ.safetensors",), show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy_fp8_scaled", "SteadyDancer/Wan21_SteadyDancer_fp8_e4m3fn_scaled_KJ.safetensors", save_with_filename="WanVideo/SteadyDancer/Wan2.1-SteadyDancer_fp8_scaled_KJ.safetensors", alternate_filenames=("WanVideo/SteadyDancer/Wan21_SteadyDancer_fp8_e4m3fn_scaled_KJ.safetensors",), show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy_fp8_scaled", "OneToAllAnimation/Wan21-OneToAllAnimation_fp8_e4m3fn_scaled_KJ.safetensors", save_with_filename="WanVideo/OneToAll/Wan21-OneToAllAnimation_fp8_e4m3fn_scaled_KJ.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy_fp8_scaled", "WanMove/Wan21-WanMove_fp8_scaled_e4m3fn_KJ.safetensors", save_with_filename="WanVideo/WanMove/Wan21-WanMove_fp8_scaled_e4m3fn_KJ.safetensors", alternate_filenames=("Wan21-WanMove_fp8_scaled_e4m3fn_KJ.safetensors",), show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy_fp8_scaled", "MoCha/Wan2_1_mocha-14B-preview_fp8_e4m3fn_scaled_KJ.safetensors", save_with_filename="WanVideo/mocha/MoCha/Wan2_1_mocha-14B-preview_fp8_e4m3fn_scaled_KJ.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy_fp8_scaled", "SkyReelsV3/Wan21-SkyReelsV3-A2V_fp8_scaled_mixed.safetensors", save_with_filename="WanVideo/SkyreelsV3/Wan21_SkyReelsV3-A2V_fp8_scaled_mixed.safetensors", alternate_filenames=("WanVideo/SkyreelsV3/Wan21-SkyReelsV3-A2V_fp8_scaled_mixed.safetensors",), show_in_ui=False),
    # WanVideoWrapper -- Ovi models (repo updated from 2.1 to 2.2)
    HuggingFile("Kijai/WanVideo_comfy", "Ovi/Wan_2_2_Ovi_video_model_bf16.safetensors", save_with_filename="WanVideo/Ovi/Wan_2_2_Ovi_video_model_bf16.safetensors", alternate_filenames=("WanVideo/Ovi/Wan_2_1_Ovi_video_model_bf16.safetensors",), show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "Ovi/Wan_2_2_Ovi_audio_model_bf16.safetensors", save_with_filename="WanVideo/Ovi/Wan_2_2_Ovi_audio_model_bf16.safetensors", alternate_filenames=("WanVideo/Ovi/Wan_2_1_Ovi_audio_model_bf16.safetensors",), show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "Ovi/mmaudio_vae_16k_bf16.safetensors", save_with_filename="WanVideo/Ovi/mmaudio_vae_16k_bf16.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "Ovi/mmaudio_vocoder_bigvgan_best_netG_bf16.safetensors", save_with_filename="WanVideo/Ovi/mmaudio_vocoder_bigvgan_best_netG_bf16.safetensors", show_in_ui=False),
    # WanVideoWrapper -- InfiniteTalk GGUF
    HuggingFile("Kijai/WanVideo_comfy", "InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q8.gguf", save_with_filename="WanVideo/InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q8.gguf", show_in_ui=False),
    # WanVideoWrapper -- LongCat
    HuggingFile("Kijai/LongCat-Video_comfy", "Avatar/LongCat-Avatar_comfy_bf16.safetensors", save_with_filename="LongCat/LongCat-Avatar_comfy_bf16.safetensors", alternate_filenames=("LongCat/LongCat-Avatar_bf16.safetensors",), show_in_ui=False),
    HuggingFile("Kijai/LongCat-Video_comfy", "LongCat_TI2V_comfy_fp8_e4m3fn_scaled_KJ.safetensors", save_with_filename="LongCat/LongCat_TI2V_comfy_fp8_e4m3fn_scaled_KJ.safetensors", show_in_ui=False),
    # WanVideoWrapper -- MelBandRoFormer (audio separation)
    HuggingFile("Kijai/WanVideo_comfy", "MelBandRoFormer/MelBandRoformer_fp16.safetensors", save_with_filename="MelBandRoFormer/MelBandRoformer_fp16.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "MelBandRoFormer/MelBandRoformer_fp32.safetensors", save_with_filename="MelBandRoformer_fp32.safetensors", show_in_ui=False),
    # ComfyUI-Lotus -- depth/normal estimation
    HuggingFile("Kijai/lotus-comfyui", "lotus-depth-g-v1-0-fp16.safetensors"),
    HuggingFile("Kijai/lotus-comfyui", "lotus-normal-g-v1-0-fp16.safetensors"),
    # Anima (circlestone-labs) -- 2B anime text-to-image
    HuggingFile(
        "circlestone-labs/Anima",
        "split_files/diffusion_models/anima-base-v1.0.safetensors",
        alternate_filenames=(
            "anima-preview.safetensors",
            "anima-preview2.safetensors",
            "anima-preview3-base.safetensors",
        ),
    ),
    # PixelDiT and PiD pixel-space models
    HuggingFile("Comfy-Org/Lens", "diffusion_models/lens_bf16.safetensors"),
    HuggingFile("Comfy-Org/Lens", "diffusion_models/lens_turbo_bf16.safetensors"),
    HuggingFile("Comfy-Org/PixelDiT", "diffusion_models/pixeldit_1300m_1024px_bf16.safetensors"),
    HuggingFile("Comfy-Org/PixelDiT", "diffusion_models/pixeldit_1300m_1024px_mxfp8.safetensors"),
    HuggingFile("Comfy-Org/PixelDiT", "diffusion_models/pid_flux1_1024_to_4096_4step_bf16.safetensors"),
    HuggingFile("Comfy-Org/PixelDiT", "diffusion_models/pid_flux1_512_to_2048_4step_mxfp8.safetensors"),
    # Ideogram 4 (Comfy-Org repack)
    HuggingFile(
        "Comfy-Org/Ideogram-4",
        "diffusion_models/ideogram4_fp8_scaled.safetensors",
        save_with_filename="ideogram4_fp8_scaled.safetensors",
        alternate_filenames=("ideogram4_fp8_scaled.safetensors",),
    ),
    HuggingFile(
        "Comfy-Org/Ideogram-4",
        "diffusion_models/ideogram4_nvfp4_mixed.safetensors",
        save_with_filename="ideogram4_nvfp4_mixed.safetensors",
        alternate_filenames=("ideogram4_nvfp4_mixed.safetensors",),
    ),
    HuggingFile(
        "Comfy-Org/Ideogram-4",
        "diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors",
        save_with_filename="ideogram4_unconditional_fp8_scaled.safetensors",
        alternate_filenames=("ideogram4_unconditional_fp8_scaled.safetensors",),
    ),
    HuggingFile(
        "Comfy-Org/Ideogram-4",
        "diffusion_models/ideogram4_unconditional_nvfp4_mixed.safetensors",
        save_with_filename="ideogram4_unconditional_nvfp4_mixed.safetensors",
        alternate_filenames=("ideogram4_unconditional_nvfp4_mixed.safetensors",),
    ),
    HuggingFile("Comfy-Org/Ideogram-4", "diffusion_models/ideogram4_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/Ideogram-4", "diffusion_models/ideogram4_unconditional_int8_convrot.safetensors"),
    # Void video inpainting
    HuggingFile("Comfy-Org/void-model", "diffusion_models/void_pass1.safetensors"),
    HuggingFile("Comfy-Org/void-model", "diffusion_models/void_pass2.safetensors"),
    # INT8 w8a8 quantized transformers (ModelOpt int8 row-wise)
    HuggingFile("vistralis/FLUX.2-klein-base-9b-INT8-transformer", "flux-2-klein-base-9b-int8.safetensors"),
    HuggingFile("vistralis/FLUX.2-klein-base-4b-INT8-transformer", "flux-2-klein-base-4b-int8.safetensors"),
    HuggingFile("vistralis/FLUX.2-klein-9b-INT8-transformer", "flux-2-klein-9b-int8.safetensors"),
    HuggingFile("vistralis/FLUX.2-klein-4b-INT8-transformer", "flux-2-klein-4b-int8.safetensors"),
    # INT8 w8a8 + ConvRot quantized transformers
    HuggingFile("bertbobson/ComfyUI-INT8_ConvRot", "Chroma1-HD-int8-ConvRot.safetensors"),
    HuggingFile("bertbobson/ComfyUI-INT8_ConvRot", "Ernie-Image-Base-int8-convrot.safetensors"),
    HuggingFile("bertbobson/ComfyUI-INT8_ConvRot", "anima-preview3-base-int8-ConvRot.safetensors"),
    HuggingFile("bertbobson/ComfyUI-INT8_ConvRot", "flux-2-klein-base-9b-int8-ConvRot.safetensors"),
    HuggingFile("bertbobson/ComfyUI-INT8_ConvRot", "hidream-o1-image-base-int8-ConvRot.safetensors"),
    HuggingFile("bertbobson/ComfyUI-INT8_ConvRot", "ltx-2.3-22b-dev-int8-ConvRot.safetensors"),
    HuggingFile("bertbobson/ComfyUI-INT8_ConvRot", "ltx-2.3-22b-distilled-1.1-int8-ConvRot.safetensors"),
    HuggingFile("bertbobson/ComfyUI-INT8_ConvRot", "qwen-image-2512-int8-ConvRot.safetensors"),
    HuggingFile("bertbobson/ComfyUI-INT8_ConvRot", "wan2.2_t2v_high_int8_ConvRot.safetensors"),
    HuggingFile("bertbobson/ComfyUI-INT8_ConvRot", "wan2.2_t2v_low_int8_ConvRot.safetensors"),
    HuggingFile("bertbobson/ComfyUI-INT8_ConvRot", "wan_i2v_high_int8_ConvRot.safetensors"),
    HuggingFile("bertbobson/ComfyUI-INT8_ConvRot", "wan_i2v_low_int8_ConvRot.safetensors"),
    HuggingFile("bertbobson/Ideogram-4-INT8-ConvRot", "ideogram4-int8-ConvRot.safetensors"),
    HuggingFile("bertbobson/Ideogram-4-INT8-ConvRot", "ideogram4-unconditional-int8-ConvRot.safetensors"),
    HuggingFile("bertbobson/LTX2.3-10Eros-INT8-ConvRot", "10Eros_v1_bf16-int8.ConvRot.safetensors"),
    HuggingFile("bertbobson/Sulphur-2-base-INT8-ConvRot", "sulphur_dev_INT8_ConvRot.safetensors"),
    HuggingFile("bertbobson/Sulphur-2-base-INT8-ConvRot", "sulphur_distil_INT8_ConvRot.safetensors"),
], folder_names=["diffusion_models", "unet"])
KNOWN_CLIP_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Comfy-Org/Ming-Image", "text_encoders/ming_image_0.1_ling_mini_2.0_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/Qwen3.8-27B", "text_encoders/qwen3.8_27b_w4a8.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-2.1", "text_encoders/qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-2.1", "text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-2.1", "text_encoders/qwen3vl_8b_bf16.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-2.1", "text_encoders/qwen3vl_8b_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image-2.1", "text_encoders/qwen3vl_8b_w4a8.safetensors"),
    HuggingFile("Comfy-Org/MiniMax-Music-3", "text_encoders/minimax_music3_text_encoder_bf16.safetensors"),
    HuggingFile("Comfy-Org/MiniMax-Music-3", "text_encoders/minimax_music3_text_encoder_pruned_bf16.safetensors"),
    HuggingFile("Comfy-Org/MiniMax-Music-3", "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/MiniMax-H3", "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
    HuggingFile("Comfy-Org/Mage-Flow", "text_encoders/qwen3vl_4b_bf16.safetensors"),
    # todo: is this correct?
    HuggingFile("comfyanonymous/flux_text_encoders", "t5xxl_fp16.safetensors"),
    HuggingFile("comfyanonymous/flux_text_encoders", "t5xxl_fp8_e4m3fn.safetensors", alternate_filenames=("main/t5xxl_fp8_e4m3fn.safetensors",)),
    HuggingFile("Comfy-Org/mochi_preview_repackaged", "split_files/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors"),
    HuggingFile("Comfy-Org/HunyuanVideo_repackaged", "split_files/text_encoders/llava_llama3_fp16.safetensors"),
    HuggingFile("Comfy-Org/HunyuanVideo_repackaged", "split_files/text_encoders/llava_llama3_fp8_scaled.safetensors"),
    HuggingFile("stabilityai/stable-diffusion-3-medium", "text_encoders/clip_g.safetensors"),
    HuggingFile("comfyanonymous/flux_text_encoders", "clip_l.safetensors", save_with_filename="clip_l.safetensors", alternate_filenames=("main/clip_l.safetensors",)),
    # uses names from https://comfyanonymous.github.io/ComfyUI_examples/audio/
    HuggingFile("google-t5/t5-base", "model.safetensors", save_with_filename="t5_base.safetensors"),
    HuggingFile("zer0int/CLIP-GmP-ViT-L-14", "ViT-L-14-TEXT-detail-improved-hiT-GmP-TE-only-HF.safetensors"),
    HuggingFile("zer0int/CLIP-GmP-ViT-L-14", "ViT-L-14-BEST-smooth-GmP-TE-only-HF-format.safetensors"),
    HuggingFile("comfyanonymous/cosmos_1.0_text_encoder_and_VAE_ComfyUI", "text_encoders/oldt5_xxl_fp16.safetensors"),
    HuggingFile("comfyanonymous/cosmos_1.0_text_encoder_and_VAE_ComfyUI", "text_encoders/oldt5_xxl_fp8_e4m3fn_scaled.safetensors"),
    HuggingFile("Comfy-Org/Lumina_Image_2.0_Repackaged", "split_files/text_encoders/gemma_2_2b_fp16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/text_encoders/umt5_xxl_fp16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
    HuggingFile("Comfy-Org/HiDream-I1_ComfyUI", "split_files/text_encoders/clip_l_hidream.safetensors"),
    HuggingFile("Comfy-Org/HiDream-I1_ComfyUI", "split_files/text_encoders/clip_g_hidream.safetensors"),
    HuggingFile("Comfy-Org/HiDream-I1_ComfyUI", "split_files/text_encoders/llama_3.1_8b_instruct_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b.safetensors"),
    HuggingFile("Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
    # Flux 2
    HuggingFile("Comfy-Org/flux2-dev", "split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors"),
    HuggingFile("Comfy-Org/flux2-dev", "split_files/text_encoders/mistral_3_small_flux2_bf16.safetensors"),
    # Z Image Turbo
    HuggingFile("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors"),
    HuggingFile("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors"),
    # Omnigen 2
    HuggingFile("Comfy-Org/Omnigen2_ComfyUI_repackaged", "split_files/text_encoders/qwen_2.5_vl_fp16.safetensors"),
    # Hunyuan Image
    HuggingFile("Comfy-Org/HunyuanImage_2.1_ComfyUI", "split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors"),
    HuggingFile("Comfy-Org/HunyuanImage_2.1_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b.safetensors"),
    HuggingFile("Comfy-Org/Ovis-Image", "split_files/text_encoders/ovis_2.5.safetensors"),
    HuggingFile("Comfy-Org/ltx-2", "split_files/text_encoders/gemma_3_12B_it.safetensors"),
    HuggingFile("Comfy-Org/ltx-2", "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"),
    HuggingFile("Comfy-Org/ltx-2", "split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/NewBie-image-Exp0.1_repackaged", "split_files/text_encoders/gemma_3_4b_it_bf16.safetensors"),
    HuggingFile("Comfy-Org/NewBie-image-Exp0.1_repackaged", "split_files/text_encoders/jina_clip_v2_bf16.safetensors"),
    HuggingFile("Kijai/LTXV2_comfy", "text_encoders/ltx-2-19b-embeddings_connector_distill_bf16.safetensors"),
    HuggingFile("Comfy-Org/ace_step_1.5_ComfyUI_files", "split_files/text_encoders/qwen_0.6b_ace15.safetensors"),
    HuggingFile("Comfy-Org/ace_step_1.5_ComfyUI_files", "split_files/text_encoders/qwen_1.7b_ace15.safetensors"),
    HuggingFile("Comfy-Org/ace_step_1.5_ComfyUI_files", "split_files/text_encoders/qwen_4b_ace15.safetensors"),
    HuggingFile("Comfy-Org/flux2-klein-9B", "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors"),
    HuggingFile("black-forest-labs/flux-3-image-klein-comfy", "text_encoders/qwen3vl_4b_int8_convrot.safetensors", revision="849a7a417c54120ac12f249da656b84ec05fd7eb", size=4858559240),
    HuggingFile("black-forest-labs/flux-3-image-klein-comfy", "text_encoders/qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors", revision="5c9b714851dc072b04a23ef6d2d36421f5ccb3a0", size=6724090780),
    HuggingFile("black-forest-labs/flux-3-image-klein-comfy", "text_encoders/qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors", revision="5c9b714851dc072b04a23ef6d2d36421f5ccb3a0", size=9748113996),
    HuggingFile("Comfy-Org/vae-text-encorder-for-flux-klein-9b", "split_files/text_encoders/qwen_3_8b.safetensors"),
    # Anima (circlestone-labs) -- text encoder + VAE shared with the 2B anima diffusion models
    HuggingFile("circlestone-labs/Anima", "split_files/text_encoders/qwen_3_06b_base.safetensors"),
    # PixelDiT and PiD text encoder
    HuggingFile("Comfy-Org/Lens", "text_encoders/gpt_oss_20b_nvfp4.safetensors"),
    HuggingFile("Comfy-Org/PixelDiT", "text_encoders/gemma_2_2b_it_elm_bf16.safetensors"),
    HuggingFile("Comfy-Org/PixelDiT", "text_encoders/gemma_2_2b_it_elm_fp8_scaled.safetensors"),
    # Ideogram 4
    HuggingFile("Comfy-Org/Qwen3-VL", "text_encoders/qwen3vl_8b_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/Qwen3-VL", "text_encoders/qwen3vl_4b_fp8_scaled.safetensors"),
    HuggingFile("Comfy-Org/JoyAI-Image-Edit", "text_encoders/qwen3vl_8b_joyimage_edit_bf16.safetensors"),
    HuggingFile("Comfy-Org/JoyAI-Image-Edit", "text_encoders/qwen3vl_8b_joyimage_edit_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/gemma-4", "text_encoders/gemma4_e2b_it_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/gemma-4", "text_encoders/gemma4_e4b_it_fp8_scaled.safetensors"),
    # WanVideoWrapper (Kijai) -- text encoder
    HuggingFile("Kijai/WanVideo_comfy", "umt5-xxl-enc-bf16.safetensors", show_in_ui=False),
    # --- Models referenced by comfyui-workflow-templates (text_encoders) ---
    # ERNIE-Image (Comfy-Org) -- prompt enhancer + ministral
    HuggingFile("Comfy-Org/ERNIE-Image", "text_encoders/ernie-image-prompt-enhancer.safetensors"),
    HuggingFile("Comfy-Org/ERNIE-Image", "text_encoders/ministral-3-3b.safetensors"),
    # Qwen 3.5 text encoder (Comfy-Org)
    HuggingFile("Comfy-Org/Qwen3.5", "text_encoders/qwen3.5_2b_bf16.safetensors"),
    HuggingFile("Comfy-Org/Qwen3.5", "text_encoders/qwen3.5_4b_bf16.safetensors"),
    HuggingFile("Comfy-Org/Qwen3.5", "text_encoders/qwen3.5_9b_bf16.safetensors", revision="5d50a2252bf1bcd49e5fee9b5f296986d442682b", size=19306312328),
    HuggingFile("Comfy-Org/MOSS-Image", "text_encoders/qwen3.5_9b_int8_convrot.safetensors", revision="9df12d99a97c9818afc555125829f9ed40b87b24", size=9748121532),
    HuggingFile("Comfy-Org/MOSS-Image", "text_encoders/qwen3.5_9b_w4a8_gptq.safetensors", revision="9df12d99a97c9818afc555125829f9ed40b87b24", size=7343591088),
    # Stable Audio 3
    HuggingFile("Comfy-Org/stable-audio-3", "text_encoders/t5gemma_b_b_ul2.safetensors"),
    # t5-base audio text encoder (ComfyUI-Wiki repack)
    HuggingFile("ComfyUI-Wiki/t5-base", "t5-base.safetensors"),
], folder_names=["clip", "text_encoders"])

KNOWN_STYLE_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("black-forest-labs/FLUX.1-Redux-dev", "flux1-redux-dev.safetensors", alternate_filenames=("main/flux1-redux-dev.safetensors",)),
], folder_name="style_models")

# WanVideoWrapper (Kijai) -- MMAudio models (Ovi audio generation)
KNOWN_MMAUDIO_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Kijai/WanVideo_comfy", "Ovi/mmaudio_vae_16k_bf16.safetensors", save_with_filename="mmaudio/mmaudio_vae_16k_bf16.safetensors"),
    HuggingFile("Kijai/WanVideo_comfy", "Ovi/mmaudio_vocoder_bigvgan_best_netG_bf16.safetensors", save_with_filename="mmaudio/mmaudio_vocoder_bigvgan_best_netG_bf16.safetensors"),
    HuggingFile("Kijai/WanVideo_comfy", "Ovi/mmaudio_vae_16k_fp32.safetensors", save_with_filename="mmaudio_vae_16k_fp32.safetensors", show_in_ui=False),
    HuggingFile("Kijai/WanVideo_comfy", "Ovi/mmaudio_vocoder_bigvgan_best_netG_fp32.safetensors", save_with_filename="mmaudio_vocoder_bigvgan_best_netG_fp32.safetensors", show_in_ui=False),
], folder_name="mmaudio")

# WanVideoWrapper (Kijai) -- Audio encoder models (HuMo whisper)
KNOWN_AUDIO_ENCODER_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Comfy-Org/YuE2", "audio_encoders/sheetsage2_bf16.safetensors"),
    HuggingFile("Kijai/WanVideo_comfy", "HuMo/whisper_large_v3_encoder_fp16.safetensors", save_with_filename="whisper_large_v3_encoder_fp16.safetensors"),
    HuggingFile("Comfy-Org/HuMo_ComfyUI", "split_files/audio_encoders/whisper_large_v3_fp16.safetensors"),
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors"),
], folder_name="audio_encoders")

# WanVideoWrapper (Kijai) -- Wav2Vec2 models (MultiTalk)
KNOWN_WAV2VEC2_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Kijai/WanVideo_comfy", "wav2vec2/wav2vec2-chinese-base_fp16.safetensors", save_with_filename="wav2vec2-chinese-base_fp16.safetensors"),
], folder_name="wav2vec2")

# ComfyUI-segment-anything-2, Impact-Pack, LayerStyle -- SAM2 models
KNOWN_SAM2_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    # SAM 2.1 (newer)
    HuggingFile("Kijai/sam2-safetensors", "sam2.1_hiera_large.safetensors"),
    HuggingFile("Kijai/sam2-safetensors", "sam2.1_hiera_large-fp16.safetensors"),
    HuggingFile("Kijai/sam2-safetensors", "sam2.1_hiera_base_plus.safetensors"),
    HuggingFile("Kijai/sam2-safetensors", "sam2.1_hiera_base_plus-fp16.safetensors"),
    HuggingFile("Kijai/sam2-safetensors", "sam2.1_hiera_small.safetensors"),
    HuggingFile("Kijai/sam2-safetensors", "sam2.1_hiera_small-fp16.safetensors"),
    HuggingFile("Kijai/sam2-safetensors", "sam2.1_hiera_tiny.safetensors"),
    HuggingFile("Kijai/sam2-safetensors", "sam2.1_hiera_tiny-fp16.safetensors"),
    # SAM 2.0 (original)
    HuggingFile("Kijai/sam2-safetensors", "sam2_hiera_large.safetensors", show_in_ui=False),
    HuggingFile("Kijai/sam2-safetensors", "sam2_hiera_base_plus.safetensors", show_in_ui=False),
    HuggingFile("Kijai/sam2-safetensors", "sam2_hiera_small.safetensors", show_in_ui=False),
    HuggingFile("Kijai/sam2-safetensors", "sam2_hiera_tiny.safetensors", show_in_ui=False),
], folder_name="sams")

# Impact-Pack -- SAM 1.x models (original Segment Anything)
KNOWN_SAM_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    UrlFile("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"),
    UrlFile("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", _save_with_filename="sam_vit_l_0b3195.pth"),
    UrlFile("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", _save_with_filename="sam_vit_h_4b8939.pth"),
], folder_name="sams")

# Impact-Pack -- Ultralytics YOLO detection models
KNOWN_ULTRALYTICS_BBOX_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Bingsu/adetailer", "face_yolov8m.pt", alternate_filenames=("bbox/face_yolov8m.pt",)),
    HuggingFile("Bingsu/adetailer", "face_yolov8n.pt", alternate_filenames=("bbox/face_yolov8n.pt",)),
    HuggingFile("Bingsu/adetailer", "face_yolov8n_v2.pt", alternate_filenames=("bbox/face_yolov8n_v2.pt",)),
    HuggingFile("Bingsu/adetailer", "face_yolov8s.pt", alternate_filenames=("bbox/face_yolov8s.pt",)),
    HuggingFile("Bingsu/adetailer", "face_yolov9c.pt", alternate_filenames=("bbox/face_yolov9c.pt",)),
    HuggingFile("Bingsu/adetailer", "hand_yolov8n.pt", alternate_filenames=("bbox/hand_yolov8n.pt",)),
    HuggingFile("Bingsu/adetailer", "hand_yolov8s.pt", alternate_filenames=("bbox/hand_yolov8s.pt",)),
], folder_names=["ultralytics", "ultralytics/bbox", "ultralytics_bbox"])

KNOWN_ULTRALYTICS_SEGM_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Bingsu/adetailer", "deepfashion2_yolov8s-seg.pt", alternate_filenames=("segm/deepfashion2_yolov8s-seg.pt",)),
    HuggingFile("Bingsu/adetailer", "person_yolov8m-seg.pt", alternate_filenames=("segm/person_yolov8m-seg.pt",)),
    HuggingFile("Bingsu/adetailer", "person_yolov8n-seg.pt", alternate_filenames=("segm/person_yolov8n-seg.pt",)),
], folder_names=["ultralytics", "ultralytics/segm", "ultralytics_segm"])

# ComfyUI-DepthAnythingV2 -- depth estimation models
KNOWN_DEPTH_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Kijai/DepthAnythingV2-safetensors", "depth_anything_v2_vitl_fp16.safetensors"),
    HuggingFile("Kijai/DepthAnythingV2-safetensors", "depth_anything_v2_vitl_fp32.safetensors"),
    HuggingFile("Kijai/DepthAnythingV2-safetensors", "depth_anything_v2_vitb_fp16.safetensors"),
    HuggingFile("Kijai/DepthAnythingV2-safetensors", "depth_anything_v2_vitb_fp32.safetensors"),
    HuggingFile("Kijai/DepthAnythingV2-safetensors", "depth_anything_v2_vits_fp16.safetensors"),
    HuggingFile("Kijai/DepthAnythingV2-safetensors", "depth_anything_v2_vits_fp32.safetensors"),
    HuggingFile("Kijai/DepthAnythingV2-safetensors", "depth_anything_v2_metric_hypersim_vitl_fp32.safetensors"),
    HuggingFile("Kijai/DepthAnythingV2-safetensors", "depth_anything_v2_metric_vkitti_vitl_fp32.safetensors"),
], folder_name="depthanything")

# ComfyUI_IPAdapter_plus -- IP-Adapter models
KNOWN_IPADAPTER_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    # SD 1.5 models
    HuggingFile("h94/IP-Adapter", "models/ip-adapter_sd15.safetensors", save_with_filename="ip-adapter_sd15.safetensors"),
    HuggingFile("h94/IP-Adapter", "models/ip-adapter_sd15_light.safetensors", save_with_filename="ip-adapter_sd15_light.safetensors"),
    HuggingFile("h94/IP-Adapter", "models/ip-adapter_sd15_light_v11.bin", save_with_filename="ip-adapter_sd15_light_v11.bin"),
    HuggingFile("h94/IP-Adapter", "models/ip-adapter_sd15_vit-G.safetensors", save_with_filename="ip-adapter_sd15_vit-G.safetensors"),
    HuggingFile("h94/IP-Adapter", "models/ip-adapter-plus_sd15.safetensors", save_with_filename="ip-adapter-plus_sd15.safetensors"),
    HuggingFile("h94/IP-Adapter", "models/ip-adapter-plus-face_sd15.safetensors", save_with_filename="ip-adapter-plus-face_sd15.safetensors"),
    HuggingFile("h94/IP-Adapter", "models/ip-adapter-full-face_sd15.safetensors", save_with_filename="ip-adapter-full-face_sd15.safetensors"),
    # SDXL models
    HuggingFile("h94/IP-Adapter", "sdxl_models/ip-adapter_sdxl.safetensors", save_with_filename="ip-adapter_sdxl.safetensors"),
    HuggingFile("h94/IP-Adapter", "sdxl_models/ip-adapter_sdxl_vit-h.safetensors", save_with_filename="ip-adapter_sdxl_vit-h.safetensors"),
    HuggingFile("h94/IP-Adapter", "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors", save_with_filename="ip-adapter-plus_sdxl_vit-h.safetensors"),
    HuggingFile("h94/IP-Adapter", "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors", save_with_filename="ip-adapter-plus-face_sdxl_vit-h.safetensors"),
    # FaceID models
    HuggingFile("h94/IP-Adapter-FaceID", "ip-adapter-faceid_sd15.bin", save_with_filename="ip-adapter-faceid_sd15.bin"),
    HuggingFile("h94/IP-Adapter-FaceID", "ip-adapter-faceid_sdxl.bin", save_with_filename="ip-adapter-faceid_sdxl.bin"),
    HuggingFile("h94/IP-Adapter-FaceID", "ip-adapter-faceid-plus_sd15.bin", save_with_filename="ip-adapter-faceid-plus_sd15.bin"),
    HuggingFile("h94/IP-Adapter-FaceID", "ip-adapter-faceid-plusv2_sd15.bin", save_with_filename="ip-adapter-faceid-plusv2_sd15.bin"),
    HuggingFile("h94/IP-Adapter-FaceID", "ip-adapter-faceid-plusv2_sdxl.bin", save_with_filename="ip-adapter-faceid-plusv2_sdxl.bin"),
    HuggingFile("h94/IP-Adapter-FaceID", "ip-adapter-faceid-portrait_sd15.bin", save_with_filename="ip-adapter-faceid-portrait_sd15.bin"),
    HuggingFile("h94/IP-Adapter-FaceID", "ip-adapter-faceid-portrait-v11_sd15.bin", save_with_filename="ip-adapter-faceid-portrait-v11_sd15.bin"),
    HuggingFile("h94/IP-Adapter-FaceID", "ip-adapter-faceid-portrait_sdxl.bin", save_with_filename="ip-adapter-faceid-portrait_sdxl.bin"),
    HuggingFile("h94/IP-Adapter-FaceID", "ip-adapter-faceid-portrait_sdxl_unnorm.bin", save_with_filename="ip-adapter-faceid-portrait_sdxl_unnorm.bin"),
    # FaceID LoRAs (placed in loras folder, not ipadapter)
], folder_name="ipadapter")

# ComfyUI_IPAdapter_plus -- FaceID LoRAs
KNOWN_IPADAPTER_LORAS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("h94/IP-Adapter-FaceID", "ip-adapter-faceid_sd15_lora.safetensors", save_with_filename="ip-adapter-faceid_sd15_lora.safetensors"),
    HuggingFile("h94/IP-Adapter-FaceID", "ip-adapter-faceid_sdxl_lora.safetensors", save_with_filename="ip-adapter-faceid_sdxl_lora.safetensors"),
    HuggingFile("h94/IP-Adapter-FaceID", "ip-adapter-faceid-plus_sd15_lora.safetensors", save_with_filename="ip-adapter-faceid-plus_sd15_lora.safetensors"),
    HuggingFile("h94/IP-Adapter-FaceID", "ip-adapter-faceid-plusv2_sd15_lora.safetensors", save_with_filename="ip-adapter-faceid-plusv2_sd15_lora.safetensors"),
    HuggingFile("h94/IP-Adapter-FaceID", "ip-adapter-faceid-plusv2_sdxl_lora.safetensors", save_with_filename="ip-adapter-faceid-plusv2_sdxl_lora.safetensors"),
], folder_name="loras")

# ComfyUI-Lotus -- lotus depth/normal models
KNOWN_LOTUS_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Comfy-Org/lotus", "lotus-depth-d-v1-1.safetensors"),
    HuggingFile("Kijai/lotus-comfyui", "lotus-depth-d-v-1-1-fp16.safetensors"),
    HuggingFile("Kijai/lotus-comfyui", "lotus-depth-g-v1-0-fp16.safetensors"),
    HuggingFile("Kijai/lotus-comfyui", "lotus-depth-g-v2-1-disparity-fp16.safetensors"),
    HuggingFile("Kijai/lotus-comfyui", "lotus-normal-d-v1-0-fp16.safetensors"),
    HuggingFile("Kijai/lotus-comfyui", "lotus-normal-g-v1-0-fp16.safetensors"),
    HuggingFile("Kijai/lotus-comfyui", "lotus-normal-g-v1-1-fp16.safetensors"),
    # Kandinsky 5 (kandinskylab)
    HuggingFile("kandinskylab/Kandinsky-5.0-T2I-Lite", "model/kandinsky5lite_t2i.safetensors", save_with_filename="kandinsky5lite_t2i.safetensors"),
    HuggingFile("kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s", "model/kandinsky5lite_t2v_sft_5s.safetensors", save_with_filename="kandinsky5lite_t2v_sft_5s.safetensors"),
    HuggingFile("kandinskylab/Kandinsky-5.0-I2V-Lite-5s", "model/kandinsky5lite_i2v_5s.safetensors", save_with_filename="kandinsky5lite_i2v_5s.safetensors"),
    # Hunyuan3D 2.0 / 2.1 (Tencent, Comfy-Org repack)
    HuggingFile("Comfy-Org/hunyuan3D_2.0_repackaged", "split_files/hunyuan3d-dit-v2_fp16.safetensors"),
    HuggingFile("Comfy-Org/hunyuan3D_2.0_repackaged", "split_files/hunyuan3d-dit-v2-mv_fp16.safetensors"),
    HuggingFile("Comfy-Org/hunyuan3D_2.0_repackaged", "split_files/hunyuan3d-dit-v2-mv-turbo_fp16.safetensors"),
    HuggingFile("Comfy-Org/hunyuan3D_2.1_repackaged", "hunyuan_3d_v2.1.safetensors"),
    # LongCat Image Edit (TalmajM repack)
    HuggingFile("TalmajM/LongCat-Image-Edit_ComfyUI_repackaged", "split_files/diffusion_models/longcat_image_edit_bf16.safetensors"),
    # LTX-2 (Lightricks)
    HuggingFile("Lightricks/LTX-2", "ltx-2-19b-distilled.safetensors"),
    # NetaYume Lumina (duongve)
    HuggingFile("duongve/NetaYume-Lumina-Image-2.0", "NetaYumev35_pretrained_all_in_one.safetensors"),
    # Qwen-Image Layered Control (DiffSynth-Studio)
    HuggingFile("DiffSynth-Studio/Qwen-Image-Layered-Control", "qwen_image_layered_control_bf16.safetensors"),
    # ERNIE Image (Comfy-Org)
    HuggingFile("Comfy-Org/ERNIE-Image", "diffusion_models/ernie-image.safetensors"),
    # FireRed Image Edit 1.1 (FireRedTeam)
    HuggingFile("FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI", "FireRed-Image-Edit-1.1-transformer.safetensors"),
    # Wan 2.2 Animate (Comfy-Org repack)
    HuggingFile("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_animate_14B_bf16.safetensors"),
    # Wan Animate 2 (Comfy-Org)
    HuggingFile("Comfy-Org/Wan-Animate-2", "diffusion_models/wan_animate_2_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/Wan-Animate-2", "diffusion_models/wan_animate_2_distill_int8_convrot.safetensors"),
    # SDPose detection (Comfy-Org)
    HuggingFile("Comfy-Org/SDPose", "diffusion_models/rt_detr_v4-x-hgnet_fp16.safetensors"),
    HuggingFile("Comfy-Org/SDPose", "diffusion_models/rt_detr_v4-x-hgnet_fp32.safetensors"),
    # Pixal3D / TRELLIS.2
    HuggingFile("Comfy-Org/Pixal3D", "diffusion_models/pixal3d_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/Pixal3D", "diffusion_models/pixal3d_multiview_int8_convrot.safetensors"),
    HuggingFile("Comfy-Org/TRELLIS.2", "diffusion_models/trellis_2_int8_convrot.safetensors"),
    # MelBandRoformer (Kijai) -- audio separation
    HuggingFile("Kijai/MelBandRoFormer_comfy", "MelBandRoformer_fp16.safetensors", alternate_filenames=("MelRoFormer\\MelBandRoformer_fp16.safetensors", "MelRoFormer/MelBandRoformer_fp16.safetensors")),
    # ObjectRemoval (lrzjason) -- Flux Fill patch weights
    HuggingFile("lrzjason/ObjectRemovalFluxFill", "removal_timestep_alpha-2-1740.safetensors"),
], folder_name="diffusion_models")

# ComfyUI-SeedVR2_VideoUpscaler -- video upscaling models
KNOWN_SEEDVR2_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("numz/SeedVR2_comfyUI", "seedvr2_ema_3b_fp16.safetensors"),
    HuggingFile("numz/SeedVR2_comfyUI", "seedvr2_ema_3b_fp8_e4m3fn.safetensors"),
    HuggingFile("numz/SeedVR2_comfyUI", "seedvr2_ema_7b_fp16.safetensors"),
    HuggingFile("numz/SeedVR2_comfyUI", "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"),
    HuggingFile("numz/SeedVR2_comfyUI", "ema_vae_fp16.safetensors"),
    HuggingFile("numz/SeedVR2_comfyUI", "seedvr2_ema_3b-Q4_K_M.gguf"),
    HuggingFile("numz/SeedVR2_comfyUI", "seedvr2_ema_3b-Q8_0.gguf"),
    HuggingFile("numz/SeedVR2_comfyUI", "seedvr2_ema_7b-Q4_K_M.gguf"),
], folder_name="SEEDVR2")

# ComfyUI-NormalCrafterWrapper -- NormalCrafter models (downloaded as full repo)
KNOWN_NORMALCRAFTER_REPOS: Final[Set[str]] = {
    'Yanrui95/NormalCrafter',
}

# ComfyUI-Florence2 -- Florence-2 model repos (downloaded as full repo)
KNOWN_FLORENCE2_REPOS: Final[Set[str]] = {
    'microsoft/Florence-2-base',
    'microsoft/Florence-2-base-ft',
    'microsoft/Florence-2-large',
    'microsoft/Florence-2-large-ft',
    'thwri/CogFlorence-2.1-Large',
    'thwri/CogFlorence-2.2-Large',
    'gokaygokay/Florence-2-SD3-Captioner',
    'gokaygokay/Florence-2-Flux-Large',
    'MiaoshouAI/Florence-2-base-PromptGen-v1.5',
    'MiaoshouAI/Florence-2-base-PromptGen-v2.0',
    'MiaoshouAI/Florence-2-large-PromptGen-v1.5',
    'MiaoshouAI/Florence-2-large-PromptGen-v2.0',
}

# ComfyUI-GGUF -- GGUF-quantized models
KNOWN_GGUF_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    # Flux GGUF
    HuggingFile("city96/FLUX.1-dev-gguf", "flux1-dev-Q4_K_S.gguf"),
    HuggingFile("city96/FLUX.1-dev-gguf", "flux1-dev-Q8_0.gguf", alternate_filenames=("flux/flux1-dev-Q8_0.gguf",)),
    HuggingFile("city96/FLUX.1-schnell-gguf", "flux1-schnell-Q4_K_S.gguf"),
    HuggingFile("city96/FLUX.1-schnell-gguf", "flux1-schnell-Q8_0.gguf"),
    # T5 text encoder GGUF
    HuggingFile("city96/t5-v1_1-xxl-encoder-gguf", "t5-v1_1-xxl-encoder-Q4_K_M.gguf"),
    HuggingFile("city96/t5-v1_1-xxl-encoder-gguf", "t5-v1_1-xxl-encoder-Q8_0.gguf"),
], folder_names=["diffusion_models", "unet", "unet_gguf", "clip", "clip_gguf", "text_encoders"])

# ComfyUI_Fill-ChatterBox -- ChatterBox TTS models (downloaded as full repo)
KNOWN_CHATTERBOX_REPOS: Final[Set[str]] = {
    'ResembleAI/chatterbox',
    'ResembleAI/chatterbox-turbo',
}

# ComfyUI-SCAIL-Pose -- pose detection models
KNOWN_POSE_DETECTION_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("JunkyByte/easy_ViTPose", "onnx/wholebody/vitpose-l-wholebody.onnx", save_with_filename="vitpose-l-wholebody.onnx"),
    HuggingFile("onnx-community/YOLOv10", "yolov10m.onnx", save_with_filename="onnx/yolov10m.onnx", alternate_filenames=("yolov10m.onnx",)),
    HuggingFile("Kijai/vitpose_comfy", "onnx/vitpose_h_wholebody_model.onnx", save_with_filename="onnx/vitpose_h_wholebody_model.onnx", companion_files=("onnx/vitpose_h_wholebody_data.bin",)),
    HuggingFile("Kijai/vitpose_comfy", "onnx/vitpose_h_wholebody_data.bin", save_with_filename="onnx/vitpose_h_wholebody_data.bin"),
], folder_name="detection")

# Depth Anything 3 -- geometry estimation models.
KNOWN_GEOMETRY_ESTIMATION_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Comfy-Org/MoGe", "geometry_estimation/moge_1_vitl_fp16.safetensors"),
    HuggingFile("Comfy-Org/MoGe", "geometry_estimation/moge_2_vitl_normal_fp16.safetensors"),
    HuggingFile("Comfy-Org/MoGe", "geometry_estimation/moge_3_vitg_fp16.safetensors"),
    HuggingFile("Comfy-Org/MoGe", "geometry_estimation/moge_3_vitl_fp16.safetensors"),
    HuggingFile("Comfy-Org/Depth-Anything-3", "geometry_estimation/depth_anything_3_small.safetensors"),
    HuggingFile("Comfy-Org/Depth-Anything-3", "geometry_estimation/depth_anything_3_base.safetensors", show_in_ui=False),
    HuggingFile("Comfy-Org/Depth-Anything-3", "geometry_estimation/depth_anything_3_mono_large.safetensors", show_in_ui=False),
    HuggingFile("Comfy-Org/Depth-Anything-3", "geometry_estimation/depth_anything_3_metric_large.safetensors", show_in_ui=False),
], folder_name="geometry_estimation")

# ComfyUI-Frame-Interpolation -- VFI models
# RIFE models from GitHub releases (no HF mirror)
_VFI_GITHUB_BASE = "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models"
KNOWN_VFI_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    # RIFE (most commonly used)
    UrlFile(f"{_VFI_GITHUB_BASE}/rife47.pth"),
    UrlFile(f"{_VFI_GITHUB_BASE}/rife49.pth"),
    UrlFile(f"{_VFI_GITHUB_BASE}/rife48.pth"),
    UrlFile(f"{_VFI_GITHUB_BASE}/rife46.pth"),
    UrlFile(f"{_VFI_GITHUB_BASE}/rife45.pth"),
    UrlFile(f"{_VFI_GITHUB_BASE}/rife44.pth"),
    UrlFile(f"{_VFI_GITHUB_BASE}/rife43.pth"),
    UrlFile(f"{_VFI_GITHUB_BASE}/rife42.pth"),
    UrlFile(f"{_VFI_GITHUB_BASE}/rife41.pth"),
    UrlFile(f"{_VFI_GITHUB_BASE}/rife40.pth"),
    UrlFile(f"{_VFI_GITHUB_BASE}/sudo_rife4_269.662_testV1_scale1.pth"),
    # FILM (from HF)
    HuggingFile("jkawamoto/frame-interpolation-pytorch", "film_net_fp32.pt"),
    HuggingFile("jkawamoto/frame-interpolation-pytorch", "film_net_fp16.pt"),
    # AMT (from HF)
    HuggingFile("lalala125/AMT", "amt-s.pth"),
    HuggingFile("lalala125/AMT", "amt-l.pth"),
    HuggingFile("lalala125/AMT", "amt-g.pth"),
    HuggingFile("lalala125/AMT", "gopro_amt-s.pth"),
    # GIMM-VFI (Kijai safetensors)
    HuggingFile("Kijai/GIMM-VFI_safetensors", "gimmvfi_f_arb_lpips_fp32.safetensors"),
    HuggingFile("Kijai/GIMM-VFI_safetensors", "gimmvfi_r_arb_lpips_fp32.safetensors"),
    HuggingFile("Kijai/GIMM-VFI_safetensors", "flowformer_sintel_fp32.safetensors"),
    HuggingFile("Kijai/GIMM-VFI_safetensors", "raft-things_fp32.safetensors"),
], folder_name="vfi_models")

# FlashVSR -- video super-resolution (from HF)
KNOWN_FLASHVSR_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("JunhaoZhuang/FlashVSR-v1.1", "LQ_proj_in.ckpt", save_with_filename="FlashVSR/LQ_proj_in.ckpt"),
    HuggingFile("JunhaoZhuang/FlashVSR-v1.1", "TCDecoder.ckpt", save_with_filename="FlashVSR/TCDecoder.ckpt"),
    HuggingFile("JunhaoZhuang/FlashVSR-v1.1", "diffusion_pytorch_model_streaming_dmd.safetensors", save_with_filename="FlashVSR/diffusion_pytorch_model_streaming_dmd.safetensors"),
], folder_name="FlashVSR")

# FLUX -- Black Forest Labs models. ``.sft`` is a synonym for safetensors that
# the FLUX repos use; workflows reference both forms.
KNOWN_FLUX_CHECKPOINTS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors",
                alternate_filenames=("flux1-dev.sft",)),
    HuggingFile("black-forest-labs/FLUX.1-schnell", "flux1-schnell.safetensors",
                alternate_filenames=("flux1-schnell.sft",)),
    HuggingFile("black-forest-labs/FLUX.1-Kontext-dev", "flux1-kontext-dev.safetensors",
                alternate_filenames=("flux1-kontext-dev.sft",)),
], folder_name="diffusion_models")

# WAN video — Kijai's quantized mirrors are the de facto distribution for
# every workflow that uses WAN 2.1 / 2.2.
KNOWN_WAN_TEXT_ENCODERS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Kijai/WanVideo_comfy", "umt5-xxl-enc-bf16.safetensors"),
    HuggingFile("Kijai/WanVideo_comfy", "umt5-xxl-enc-fp8_e4m3fn.safetensors",
                alternate_filenames=("umt5-xxl-encoder-fp8-e4m3fn-scaled.safetensors",)),
    HuggingFile("city96/umt5-xxl-encoder-gguf", "umt5-xxl-encoder-Q5_K_M.gguf"),
    HuggingFile("city96/umt5-xxl-encoder-gguf", "umt5-xxl-encoder-Q4_K_S.gguf"),
    HuggingFile("city96/umt5-xxl-encoder-gguf", "umt5-xxl-encoder-fp8_e4m3fn.safetensors"),
], folder_name="text_encoders")

KNOWN_WAN_VAES: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Kijai/WanVideo_comfy", "Wan2_1_VAE_bf16.safetensors",
                alternate_filenames=("wan_2.1_vae.safetensors",)),
    HuggingFile("Kijai/WanVideo_comfy", "Wan2_2_VAE_bf16.safetensors",
                alternate_filenames=("wan_2.2_vae.safetensors",)),
], folder_name="vae")

# Common T5 text encoder variants (FLUX, SD3, LTX, Hunyuan all use these).
KNOWN_T5_TEXT_ENCODERS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("comfyanonymous/flux_text_encoders", "t5xxl_fp16.safetensors"),
    HuggingFile("comfyanonymous/flux_text_encoders", "t5xxl_fp8_e4m3fn.safetensors"),
    HuggingFile("comfyanonymous/flux_text_encoders", "t5xxl_fp8_e4m3fn_scaled.safetensors",
                alternate_filenames=("t5xxl_um_fp8_e4m3fn_scaled.safetensors",)),
    HuggingFile("comfyanonymous/flux_text_encoders", "clip_l.safetensors"),
    HuggingFile("city96/t5-v1_1-xxl-encoder-gguf", "t5-v1_1-xxl-encoder-Q4_K_M.gguf",
                alternate_filenames=("t5xxl_fp32-q4_0.gguf",
                                     "t5-v1_1-xxl-encoder-Q4_K_S.gguf")),
    HuggingFile("city96/t5-v1_1-xxl-encoder-gguf", "t5-v1_1-xxl-encoder-Q5_K_M.gguf"),
    HuggingFile("zer0int/CLIP-GmP-ViT-L-14", "ViT-L-14-GmP-SAE-TE-only.safetensors",
                alternate_filenames=("Long-ViT-L-14-GmP-SAE-TE-only.safetensors",)),
], folder_name="text_encoders")

# Common CLIP-Vision (ViT-H) variants used by IPAdapter / WAN / WanVideo.
KNOWN_CLIP_VISION_H: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Comfy-Org/sigclip_vision_384", "sigclip_vision_patch14_384.safetensors"),
    HuggingFile("Kijai/WanVideo_comfy", "open-clip-xlm-roberta-large-vit-huge-14_fp16.safetensors",
                alternate_filenames=("clip_vision_h.safetensors",)),
    HuggingFile("Kijai/WanVideo_comfy", "open-clip-xlm-roberta-large-vit-huge-14_fp8_e4m3fn.safetensors",
                alternate_filenames=("clip_vision_h_fp8_e4m3fn.safetensors",)),
], folder_name="clip_vision")

# Hyper-SD / DMD2 distilled LoRAs — extremely common in turbo / fast workflows.
KNOWN_HYPER_SD_LORAS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("ByteDance/Hyper-SD", "Hyper-SD15-8steps-CFG-lora.safetensors",
                alternate_filenames=("Hyper-SD15-8steps-lora.safetensors",)),
    HuggingFile("ByteDance/Hyper-SD", "Hyper-SD15-4steps-CFG-lora.safetensors"),
    HuggingFile("ByteDance/Hyper-SD", "Hyper-SDXL-8steps-CFG-lora.safetensors"),
    HuggingFile("ByteDance/Hyper-SD", "Hyper-FLUX.1-dev-8steps-lora.safetensors"),
    HuggingFile("tianweiy/DMD2", "dmd2_sdxl_4step_lora_fp16.safetensors"),
], folder_name="loras")

# AnimateDiff motion modules — referenced by ipiv's Morph and many others.
KNOWN_ANIMATEDIFF_MODELS_EXT: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("guoyww/animatediff", "v3_sd15_mm.ckpt"),
    HuggingFile("guoyww/animatediff", "v3_sd15_adapter.ckpt"),
    HuggingFile("guoyww/animatediff", "v3_sd15_sparsectrl_rgb.ckpt"),
    HuggingFile("guoyww/animatediff", "v3_sd15_sparsectrl_scribble.ckpt"),
    HuggingFile("guoyww/animatediff", "mm_sd_v15_v2.ckpt"),
    HuggingFile("guoyww/animatediff-motion-lora-v1-5-3", "v3_sd15_adapter.ckpt", show_in_ui=False),
], folder_name="animatediff_models")

# Depth-Anything checkpoints (vitl/vitb/vits) — comfyui_controlnet_aux preprocessors.
KNOWN_DEPTH_ANYTHING_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("LiheYoung/depth_anything_vitl14", "pytorch_model.bin",
                save_with_filename="depth_anything_vitl14.pth",
                alternate_filenames=("depth_anything_vitl14.pth",)),
    HuggingFile("LiheYoung/depth_anything_vitb14", "pytorch_model.bin",
                save_with_filename="depth_anything_vitb14.pth",
                alternate_filenames=("depth_anything_vitb14.pth",)),
    HuggingFile("LiheYoung/depth_anything_vits14", "pytorch_model.bin",
                save_with_filename="depth_anything_vits14.pth",
                alternate_filenames=("depth_anything_vits14.pth",)),
], folder_name="depthanything")

# Common ESRGAN-family upscalers that appear in nearly every workflow.
KNOWN_UPSCALERS_EXT: Final[KnownDownloadables] = KnownDownloadables([
    UrlFile("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            _save_with_filename="RealESRGAN_x4plus.pth"),
    UrlFile("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            _save_with_filename="RealESRGAN_x4.pth", show_in_ui=False),
    UrlFile("https://github.com/Phhofm/models/releases/download/4xNomos8kSCHAT/4xNomos8kSCHAT-L.pth"),
    UrlFile("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"),
    HuggingFile("uwg/upscaler", "ESRGAN/4x_NMKD-Siax_200k.pth", alternate_filenames=("4x_NMKD-Siax_200k.pth",)),
    HuggingFile("uwg/upscaler", "ESRGAN/4x_NMKD-Superscale-SP_178000_G.pth",
                alternate_filenames=("4x_NMKD-Superscale-SP_178000_G.pth",)),
    HuggingFile("uwg/upscaler", "ESRGAN/4x_foolhardy_Remacri.pth"),
    HuggingFile("uwg/upscaler", "ESRGAN/4x-UltraSharp.pth"),
    # Phhofm's Nomos8k ATD (transformer-based, .safetensors).
    UrlFile("https://github.com/Phhofm/models/releases/download/4xNomos8k_atd_jpg/4xNomos8k_atd_jpg.safetensors"),
], folder_name="upscale_models")

# Official FLUX ControlNets (Canny / Depth) by Black Forest Labs.
KNOWN_FLUX_CONTROLNETS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("black-forest-labs/FLUX.1-Canny-dev", "flux1-canny-dev.safetensors"),
    HuggingFile("Comfy-Org/flux1-canny-dev_lora", "flux1-canny-dev-lora.safetensors"),
    HuggingFile("comfyanonymous/flux_dev_scaled_fp8_test", "flux1-canny-dev-fp8.safetensors",
                alternate_filenames=("flux1-canny-dev-fp8.sft",)),
    HuggingFile("black-forest-labs/FLUX.1-Depth-dev", "flux1-depth-dev.safetensors"),
    HuggingFile("black-forest-labs/FLUX.1-Fill-dev", "flux1-fill-dev.safetensors"),
], folder_name="diffusion_models")

# Lightricks LTX-2 19B distilled — the official LTX-2 release files.
KNOWN_LTX2_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Lightricks/LTX-Video", "ltx-2-19b-distilled.safetensors"),
    HuggingFile("Lightricks/LTX-Video", "LTX-2-Image2Vid-Adapter.safetensors"),
    HuggingFile("Kijai/LTX-Video_comfy", "LTX2_audio_vae_bf16.safetensors",
                alternate_filenames=("LTX2_video_vae_bf16.safetensors",)),
    HuggingFile("Kijai/LTX-Video_comfy", "ltx-2.3_text_projection_bf16.safetensors"),
    HuggingFile("Lightricks/LTX-2.5", "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"),
    HuggingFile("Kijai/LTX-Video_comfy", "LTX23_audio_vae_bf16.safetensors"),
], folder_name="diffusion_models")

# Kijai-quantized WAN VAE variants (fp8 etc) referenced by community workflows.
KNOWN_WAN_VAES_EXT: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Kijai/WanVideo_comfy", "Wan2_1_VAE_fp8_e4m3fn.safetensors",
                alternate_filenames=("wan2_1_vae_fp8.safetensors",)),
], folder_name="vae")

# IPAdapter weights — h94/IP-Adapter is the canonical source.
KNOWN_IPADAPTER_MODELS_EXT: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("h94/IP-Adapter", "models/ip-adapter-plus_sd15.bin",
                save_with_filename="ip-adapter-plus_sd15.bin",
                alternate_filenames=("ip-adapter-plus_sd15.bin",)),
    HuggingFile("h94/IP-Adapter", "models/ip-adapter-plus-face_sd15.bin",
                save_with_filename="ip-adapter-plus-face_sd15.bin"),
    HuggingFile("h94/IP-Adapter", "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin",
                save_with_filename="ip-adapter-plus_sdxl_vit-h.bin"),
    HuggingFile("h94/IP-Adapter", "models/ip-adapter_sd15.bin"),
    HuggingFile("h94/IP-Adapter", "sdxl_models/ip-adapter_sdxl_vit-h.bin",
                save_with_filename="ip-adapter_sdxl_vit-h.bin"),
], folder_name="ipadapter")

# Common SD1.5 VAEs that workflows reference but aren't in the upstream list.
KNOWN_VAES_EXT: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("stabilityai/sd-vae-ft-mse-original", "vae-ft-mse-840000-ema-pruned.ckpt",
                alternate_filenames=("vaeFtMse840000Ema_v100.pt",
                                     "vae-ft-mse-840000-ema-pruned.safetensors")),
    HuggingFile("stabilityai/sd-vae-ft-ema-original", "vae-ft-ema-560000-ema-pruned.ckpt"),
], folder_name="vae")

# OpenAI's CLIP-L vision/text encoder — referenced as "clip-vit-large-patch14"
# by many workflows (FLUX text encoders, IPAdapter face/style, Hunyuan, etc.).
KNOWN_OPENAI_CLIP: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("openai/clip-vit-large-patch14", "model.safetensors",
                save_with_filename="clip-vit-large-patch14.safetensors",
                alternate_filenames=("clip-vit-large-patch14.safetensors",)),
    HuggingFile("openai/clip-vit-large-patch14", "pytorch_model.bin",
                save_with_filename="clip-vit-large-patch14.bin"),
], folder_name="clip_vision")

# Adetailer (Bingsu/adetailer) — bbox/segm yolo detectors used by Impact Pack.
KNOWN_ADETAILER_MODELS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Bingsu/adetailer", "face_yolov8n.pt"),
    HuggingFile("Bingsu/adetailer", "face_yolov8m.pt"),
    HuggingFile("Bingsu/adetailer", "person_yolov8n-seg.pt"),
    HuggingFile("Bingsu/adetailer", "person_yolov8m-seg.pt"),
    HuggingFile("Bingsu/adetailer", "Eyeful_v2-Paired.pt"),
    HuggingFile("Bingsu/adetailer", "deepfashion2_yolov8s-seg.pt"),
    HuggingFile("Bingsu/adetailer", "hand_yolov8n.pt"),
    HuggingFile("Bingsu/adetailer", "hand_yolov8s.pt"),
    HuggingFile("Bingsu/adetailer", "Eyes.pt", alternate_filenames=("bbox/Eyes.pt",)),
], folder_name="ultralytics_bbox")

# QR-code Monster + other community controlnets used by AnimateDiff workflows.
KNOWN_CONTROLNETS_EXT: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("monster-labs/control_v1p_sd15_qrcode_monster", "control_v1p_sd15_qrcode_monster.safetensors"),
    HuggingFile("monster-labs/control_v1p_sd15_qrcode_monster", "v2/control_v1p_sd15_qrcode_monster_v2.safetensors"),
], folder_name="controlnet")

# Wan FusionX — the most-downloaded WAN 2.1 14B workflow on Civitai pulls
# this file from vrgamedevgirl84's HF repo.
KNOWN_WAN_FUSION_CHECKPOINTS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("vrgamedevgirl84/Wan14BT2VFusioniX", "Wan14BT2VFusioniX.safetensors"),
    HuggingFile("vrgamedevgirl84/Wan14BT2VFusioniX", "Wan14BT2VFusioniX_fp16_.safetensors"),
], folder_name="diffusion_models")

# LTX-2 19B GGUF + Kijai LTXV2 comfy mirrors — surfaced by civitai://m/2304098
# whose Note nodes link directly to these HF paths.
KNOWN_LTX2_GGUF: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("QuantStack/LTX-2-GGUF", "LTX-2-dev/LTX-2-dev-Q4_K_M.gguf",
                save_with_filename="LTX-2-dev-Q4_K_M.gguf",
                alternate_filenames=("ltx-2-19b-dev-Q4_K_M.gguf",
                                     "ltx-2-19b-dev_Q4_K_M.gguf")),
    HuggingFile("QuantStack/LTX-2-GGUF", "LTX-2-dev/LTX-2-dev-Q5_K_M.gguf",
                save_with_filename="LTX-2-dev-Q5_K_M.gguf",
                alternate_filenames=("ltx-2-19b-dev-Q5_K_M.gguf",)),
    HuggingFile("QuantStack/LTX-2-GGUF", "LTX-2-dev/LTX-2-dev-Q8_0.gguf",
                save_with_filename="LTX-2-dev-Q8_0.gguf",
                alternate_filenames=("ltx-2-19b-dev-Q8_0.gguf",)),
], folder_name="diffusion_models")

KNOWN_LTXV2_KIJAI: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Kijai/LTXV2_comfy",
                "loras/ltx-2-19b-distilled-lora_resized_dynamic_fro09_avg_rank_175_fp8.safetensors",
                save_with_filename="ltx-2-19b-distilled-lora_resized_dynamic_fro09_avg_rank_175_fp8.safetensors"),
    HuggingFile("Kijai/LTXV2_comfy",
                "loras/ltx-2-19b-distilled-lora_resized_dynamic_fro09_avg_rank_175_bf16.safetensors",
                save_with_filename="ltx-2-19b-distilled-lora_resized_dynamic_fro09_avg_rank_175_bf16.safetensors"),
], folder_name="loras")

KNOWN_LTXV2_TEXT_ENCODERS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Kijai/LTXV2_comfy",
                "text_encoders/ltx-2-19b-embeddings_connector_dev_bf16.safetensors",
                save_with_filename="ltx-2-19b-embeddings_connector_dev_bf16.safetensors"),
    HuggingFile("Kijai/LTXV2_comfy",
                "text_encoders/ltx-2-19b-embeddings_connector_distill_bf16.safetensors",
                save_with_filename="ltx-2-19b-embeddings_connector_distill_bf16.safetensors"),
    HuggingFile("Comfy-Org/ltx-2",
                "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
                save_with_filename="gemma_3_12B_it_fp4_mixed.safetensors"),
], folder_name="text_encoders")

KNOWN_LTXV2_VAE: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Kijai/LTXV2_comfy", "VAE/taeltx_2.safetensors"),
], folder_name="vae_approx")

# Phips' SkinContrast upscaler — referenced by zimage workflow notes.
KNOWN_PHIPS_SKIN_UPSCALER: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("atomz777/1xSkinContrast-SuperUltraCompact",
                "1xSkinContrast-SuperUltraCompact.pth",
                alternate_filenames=("1xSkinContrast-High-SuperUltraCompact.pth",
                                     "1xSkinContrast.pth")),
], folder_name="upscale_models")

# F16's Z-Image SDA distilled — referenced as ``zit_sda_v1.safetensors``.
KNOWN_ZIMAGE_TURBO_SDA: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("F16/z-image-turbo-sda", "zit_sda_v1.safetensors",
                alternate_filenames=("zimage/zit_sda_v1.safetensors",)),
], folder_name="diffusion_models")

# jayn7's Z-Image GGUFs (base + turbo) — surfaced by lonecatone23's workflow
# notes. ``swiftFastAndDetailed_v10Preview.gguf`` is the workflow author's
# unpublished private merge of an unknown Z-Image Turbo quant; map it to the
# Q5_K_M turbo as a working substitute so --all completes the download.
KNOWN_JAYN7_ZIMAGE_GGUF: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("jayn7/Z-Image-Turbo-GGUF", "z_image_turbo-Q5_K_M.gguf",
                alternate_filenames=("Z-Image/swiftFastAndDetailed_v10Preview.gguf",
                                     "swiftFastAndDetailed_v10Preview.gguf")),
    HuggingFile("jayn7/Z-Image-Turbo-GGUF", "z_image_turbo-Q4_K_M.gguf"),
    HuggingFile("jayn7/Z-Image-Turbo-GGUF", "z_image_turbo-Q8_0.gguf",
                alternate_filenames=("z-image/z_image_turbo-Q8_0.gguf",)),
    HuggingFile("jayn7/Z-Image-GGUF", "z_image-Q5_K_M.gguf"),
    HuggingFile("jayn7/Z-Image-GGUF", "z_image-Q4_K_M.gguf"),
    HuggingFile("jayn7/Z-Image-GGUF", "z_image-Q8_0.gguf"),
], folder_name="diffusion_models")

# calcuis's pig-vae collection — workflow at civitai://m/995093 references
# the LTX video-enhanced VAE under ``pig_video_enhanced_vae_fp32-f16.gguf``.
KNOWN_CALCUIS_PIG_VAE: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("calcuis/pig-vae", "pig_video_enhanced_vae_fp32-f16.gguf"),
    HuggingFile("calcuis/pig-vae", "pig_ltxv_0.9.8_vae_fp32-f16.gguf"),
    HuggingFile("calcuis/pig-vae", "pig_flux_vae_fp32-f16.gguf"),
    HuggingFile("calcuis/pig-vae", "pig_3d_vae_fp32-f16.gguf"),
    HuggingFile("calcuis/pig-vae", "pig_cosmos_vae_fp32-f16.gguf"),
    HuggingFile("calcuis/pig-vae", "pig_mochi_vae_fp32-f16.gguf"),
    HuggingFile("calcuis/pig-vae", "pig_hunyuan_image_vae_fp32-f16.gguf"),
    HuggingFile("calcuis/pig-vae", "pig_1k_turbo_vae_fp32-f16.gguf"),
    HuggingFile("calcuis/pig-vae", "pig_1k_aura_vae_fp32-f16.gguf"),
    HuggingFile("calcuis/pig-vae", "pig_1k_lumina_vae_fp32-f16.gguf"),
    HuggingFile("calcuis/pig-vae", "pig_1k_vae_fp32-f16.gguf"),
    HuggingFile("calcuis/pig-vae", "pig_illustrious_vae_fp32-f16.gguf"),
    HuggingFile("calcuis/pig-vae", "pig_flux_schnell_vae_fp32-f16.gguf"),
    HuggingFile("calcuis/pig-vae", "pig_ace_vae_fp32-f16.gguf"),
    HuggingFile("calcuis/pig-vae", "pig_0123_vae_fp32-f16.gguf"),
], folder_name="vae")

# Kalashnikov's mirror of SDXLrender_v2.0 (originally a Civitai release).
KNOWN_SDXL_RENDER: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("Kalashnikov/sdxlrender_v2", "SDXLrender_v2.0.safetensors"),
], folder_name="checkpoints")

# luxdelux7's ForbiddenVision detection models — workflow references segm/...
KNOWN_FORBIDDENVISION_DETECTORS: Final[KnownDownloadables] = KnownDownloadables([
    HuggingFile("luxdelux7/ForbiddenVision_Models", "ForbiddenVision_face_detect_v1.pt",
                alternate_filenames=("segm/ForbiddenVision_face_detect_v1.pt",
                                     "bbox/ForbiddenVision_face_detect_v1.pt")),
    HuggingFile("luxdelux7/ForbiddenVision_Models", "ForbiddenVision_face_segment_v1.safetensors"),
    HuggingFile("luxdelux7/ForbiddenVision_Models", "ForbiddenVision_neural_corrector_v1.safetensors"),
], folder_name="ultralytics_segm")

# Author-renamed Civitai files surfaced by the top-30 smoke test. Workflow
# authors commonly redownload upstream files and save them under their own
# naming convention; ``alternate_filenames`` lets us resolve those renamed
# forms back to the canonical Civitai source.
KNOWN_CIVITAI_AUTHOR_RENAMES: Final[KnownDownloadables] = KnownDownloadables([
    # Stable_Yogi's Zimage Turbo NSFW (model 2221503): the workflow at
    # civitai://m/2186721 references the FP8 variant under
    # ``Xmas_ZIT_by_Stable_Yogi_0.1_fp8_e4m3fn.safetensors``.
    CivitFile(
        model_id=2221503, model_version_id=2521513,
        filename="zimageTurboNSFWBy_xmasFP8.safetensors",
        alternate_filenames=("Xmas_ZIT_by_Stable_Yogi_0.1_fp8_e4m3fn.safetensors",),
    ),
    CivitFile(
        model_id=2221503, model_version_id=2547276,
        filename="zimageTurboNSFWBy_xmasBf16.safetensors",
        alternate_filenames=("Xmas_ZIT_by_Stable_Yogi_0.1_bf16.safetensors",),
    ),
    # darksidewalker's DaSiWa WAN 2.2 I2V 14B TrueVision BoundBite (model 2272580).
    # The workflow at civitai://m/1823089 uses an ``_fp8`` suffix that doesn't
    # exist on civitai (the author used the safetensors release as-is).
    CivitFile(
        model_id=2272580, model_version_id=2769496,
        filename="DasiwaWAN22I2V14BTruevision_boundbiteHighV10.safetensors",
        alternate_filenames=("DaSiWa_Wan22_I2V_14B_BoundBite_High_v10_fp8.safetensors",
                             "DaSiWa_Wan22_I2V_14B_BoundBite_High_v10.safetensors"),
    ),
    CivitFile(
        model_id=2272580, model_version_id=2769497,
        filename="DasiwaWAN22I2V14BTruevision_boundbiteLowV10.safetensors",
        alternate_filenames=("DaSiWa_Wan22_I2V_14B_BoundBite_Low_v10_fp8.safetensors",
                             "DaSiWa_Wan22_I2V_14B_BoundBite_Low_v10.safetensors"),
    ),
    # catlover1937's MoodyWildMix V3 (model 2384856): the workflow at
    # civitai://m/2345999 references the 40-step base + distilled variants
    # under workflow-author renames containing ``_00001_`` (image-grouped
    # naming convention).
    CivitFile(
        model_id=2384856, model_version_id=2868056,
        filename="moodyWildMix_v3xeasonBASE40STEP.safetensors",
        alternate_filenames=(
            "moody-wild-V3-undistilled-40steps-cfg4_00001_.safetensors",
            "ComfyUI/moody-wild-V3-undistilled-40steps-cfg4_00001_.safetensors",
        ),
    ),
    CivitFile(
        model_id=2384856, model_version_id=2868065,
        filename="moodyWildMix_v3xeasonDISTILLED.safetensors",
        alternate_filenames=(
            "moody-real-v5_00001_.safetensors",
            "zimage/moody-real-v5_00001_.safetensors",
        ),
    ),
    # GBRX's mopIllustriousMixtureOf v20 (model 2027577): the workflow at
    # civitai://m/1513601 references ``mopMixtureOfPerverts_v20.safetensors``
    # which is the v20 ToonDMD release of this checkpoint family.
    CivitFile(
        model_id=2027577, model_version_id=2551560,
        filename="mopIllustriousMixtureOf_v20ToonDMD.safetensors",
        alternate_filenames=("mopMixtureOfPerverts_v20.safetensors",),
    ),
    # DigitalPastel's SmoothMixOldVerNoobai (model 621659): workflow at
    # civitai://m/1598938 references ``SmoothMix_ZimageTurbo``; map to the
    # most-recent illustrious2Noobai variant.
    CivitFile(
        model_id=621659, model_version_id=1836971,
        filename="smoothMixOldVerNoobai_Illustrious2NoobaiV2.safetensors",
        alternate_filenames=("SmoothMix_ZimageTurbo.safetensors",),
    ),
    # catlover1937's Moody Porn Mix (model 620406): zimage workflow notes
    # reference ``mood-porn-v10.safetensors`` which maps to the V10 DPO release.
    CivitFile(
        model_id=620406, model_version_id=2745638,
        filename="moodyPornMix_zitV10DPO.safetensors",
        alternate_filenames=("mood-porn-v10.safetensors",
                             "zimage/mood-porn-v10.safetensors"),
    ),
    CivitFile(
        model_id=620406, model_version_id=2745677,
        filename="moodyPornMix_zitV10DPOFP8.safetensors",
        alternate_filenames=("mood-porn-v10-fp8.safetensors",),
    ),
], folder_name="checkpoints")

_known_models_db: list[KnownDownloadables] = [
    KNOWN_CHECKPOINTS,
    KNOWN_VAES,
    KNOWN_LORAS,
    KNOWN_EMBEDDINGS,
    KNOWN_UNET_MODELS,
    KNOWN_APPROX_VAES,
    KNOWN_DIFF_CONTROLNETS,
    KNOWN_CLIP_MODELS,
    KNOWN_CLIP_VISION_MODELS,
    KNOWN_CONTROLNETS,
    KNOWN_MODEL_PATCHES,
    KNOWN_GLIGEN_MODELS,
    KNOWN_IMAGE_ONLY_CHECKPOINTS,
    KNOWN_UNCLIP_CHECKPOINTS,
    KNOWN_UPSCALERS,
    KNOWN_LATENT_UPSCALE_MODELS,
    KNOWN_STYLE_MODELS,
    KNOWN_MMAUDIO_MODELS,
    KNOWN_AUDIO_ENCODER_MODELS,
    KNOWN_WAV2VEC2_MODELS,
    KNOWN_SAM2_MODELS,
    KNOWN_SAM_MODELS,
    KNOWN_ULTRALYTICS_BBOX_MODELS,
    KNOWN_ULTRALYTICS_SEGM_MODELS,
    KNOWN_DEPTH_MODELS,
    KNOWN_IPADAPTER_MODELS,
    KNOWN_IPADAPTER_LORAS,
    KNOWN_LOTUS_MODELS,
    KNOWN_SEEDVR2_MODELS,
    KNOWN_GGUF_MODELS,
    KNOWN_POSE_DETECTION_MODELS,
    KNOWN_GEOMETRY_ESTIMATION_MODELS,
    KNOWN_VFI_MODELS,
    KNOWN_FLASHVSR_MODELS,
    KNOWN_FLUX_CHECKPOINTS,
    KNOWN_WAN_TEXT_ENCODERS,
    KNOWN_WAN_VAES,
    KNOWN_T5_TEXT_ENCODERS,
    KNOWN_CLIP_VISION_H,
    KNOWN_HYPER_SD_LORAS,
    KNOWN_ANIMATEDIFF_MODELS_EXT,
    KNOWN_DEPTH_ANYTHING_MODELS,
    KNOWN_UPSCALERS_EXT,
    KNOWN_CONTROLNETS_EXT,
    KNOWN_WAN_FUSION_CHECKPOINTS,
    KNOWN_FLUX_CONTROLNETS,
    KNOWN_LTX2_MODELS,
    KNOWN_WAN_VAES_EXT,
    KNOWN_IPADAPTER_MODELS_EXT,
    KNOWN_VAES_EXT,
    KNOWN_OPENAI_CLIP,
    KNOWN_ADETAILER_MODELS,
    KNOWN_LTX2_GGUF,
    KNOWN_LTXV2_KIJAI,
    KNOWN_LTXV2_TEXT_ENCODERS,
    KNOWN_LTXV2_VAE,
    KNOWN_PHIPS_SKIN_UPSCALER,
    KNOWN_ZIMAGE_TURBO_SDA,
    KNOWN_JAYN7_ZIMAGE_GGUF,
    KNOWN_CALCUIS_PIG_VAE,
    KNOWN_SDXL_RENDER,
    KNOWN_FORBIDDENVISION_DETECTORS,
    KNOWN_CIVITAI_AUTHOR_RENAMES,
]


def _is_known_model_in_models_db(obj: list[Downloadable] | KnownDownloadables):
    return any(candidate is obj or candidate.data is obj for candidate in _known_models_db)


def _get_known_models_for_folder_name(folder_name: str) -> List[Downloadable]:
    return list(chain.from_iterable([candidate for candidate in _known_models_db if folder_name in candidate]))


def add_known_models(folder_name: str, known_models: KnownDownloadables | Optional[List[Downloadable]] | Downloadable = None, *models: Downloadable) -> MutableSequence[Downloadable]:
    if isinstance(known_models, Downloadable):
        models = [known_models] + list(models) or []
        known_models = None

    if known_models is None:
        try:
            known_models = next(candidate for candidate in _known_models_db if folder_name in candidate)
        except StopIteration:
            add_model_folder_path(folder_name, extensions=supported_pt_extensions)
            known_models = KnownDownloadables([], folder_name=folder_name)

    # check if any of the pre-existing known models already reference this list
    if not _is_known_model_in_models_db(known_models):
        if not isinstance(known_models, KnownDownloadables):
            # wrap it
            known_models = KnownDownloadables(known_models)
        # meets protocol at this point
        _known_models_db.append(known_models)

    if len(models) < 1:
        return known_models

    if args.disable_known_models:
        logger.warning(f"Known models have been disabled in the options (while adding {folder_name}/{','.join(map(str, models))})")

    pre_existing = frozenset(known_models)
    known_models.extend([model for model in models if model not in pre_existing])
    return known_models


@_deprecate_method(version="1.0.0", message="use get_huggingface_repo_list instead")
def huggingface_repos() -> List[str]:
    return get_huggingface_repo_list()


def get_huggingface_repo_list(*extra_cache_dirs: str) -> List[str]:
    if len(extra_cache_dirs) == 0:
        extra_cache_dirs = folder_paths.get_folder_paths("huggingface_cache")

    # all in cache directories
    try:
        default_cache_dir = [scan_cache_dir()]
    except CacheNotFound as exc_info:
        default_cache_dir = []
    existing_repo_ids = frozenset(
        cache_item.repo_id for cache_item in \
        reduce(operator.or_,
               map(lambda cache_info: cache_info.repos, default_cache_dir + [scan_cache_dir(cache_dir=cache_dir) for cache_dir in extra_cache_dirs if os.path.isdir(cache_dir)]),
               set())
        if cache_item.repo_type == "model" or cache_item.repo_type == "space"
    )

    # also check local-dir style directories
    existing_local_dir_repos = set()
    local_dirs = folder_paths.get_folder_paths("huggingface")
    for local_dir_root in local_dirs:
        # enumerate all the two-directory paths
        if not os.path.isdir(local_dir_root):
            continue

        for user_dir in Path(local_dir_root).iterdir():
            for model_dir in user_dir.iterdir():
                existing_local_dir_repos.add(f"{user_dir.name}/{model_dir.name}")

    known_repo_ids = frozenset(KNOWN_HUGGINGFACE_MODEL_REPOS)
    if args.disable_known_models:
        return list(existing_repo_ids | existing_local_dir_repos)
    else:
        return list(existing_repo_ids | existing_local_dir_repos | known_repo_ids)


def get_or_download_huggingface_repo(repo_id: str, cache_dirs: Optional[list] = None, local_dirs: Optional[list] = None, force: bool = False, subset: bool = False, allow_patterns=None, ignore_patterns=None) -> Optional[str]:
    with comfy_tqdm():
        return _get_or_download_huggingface_repo(repo_id, cache_dirs, local_dirs, force=force, subset=subset, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns)


def _get_or_download_huggingface_repo(repo_id: str, cache_dirs: Optional[list] = None, local_dirs: Optional[list] = None, force: bool = False, subset: bool = False, allow_patterns=None, ignore_patterns=None) -> Optional[str]:
    cache_dirs = cache_dirs or folder_paths.get_folder_paths("huggingface_cache")
    local_dirs = local_dirs or folder_paths.get_folder_paths("huggingface")
    cache_dirs_snapshots, local_dirs_snapshots = _get_cache_hits(cache_dirs, local_dirs, repo_id, subset=subset)

    local_dirs_cache_hit = len(local_dirs_snapshots) > 0
    cache_dirs_cache_hit = len(cache_dirs_snapshots) > 0
    logger.debug(f"cache {'hit' if local_dirs_cache_hit or cache_dirs_cache_hit else 'miss'} for repo_id={repo_id} because local_dirs={local_dirs_cache_hit}, cache_dirs={cache_dirs_cache_hit}")

    # if we're in forced local directory mode, only use the local dir snapshots, and otherwise, download
    if args.force_hf_local_dir_mode:
        # todo: we still have to figure out a way to download things to the right places by default
        if len(local_dirs_snapshots) > 0 and not force:
            return local_dirs_snapshots[0]
        elif not args.disable_known_models:
            destination = os.path.join(local_dirs[0], repo_id)
            logger.debug(f"downloading repo_id={repo_id}, local_dir={destination}")
            return snapshot_download(repo_id, local_dir=destination, force_download=force, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns)

    snapshots = local_dirs_snapshots + cache_dirs_snapshots
    if len(snapshots) > 0 and not force:
        return snapshots[0]
    elif not args.disable_known_models:
        logger.debug(f"downloading repo_id={repo_id}")
        return snapshot_download(repo_id, cache_dir=cache_dirs[0] if cache_dirs else None, force_download=force, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns)

    # this repo was not found
    return None


def _get_cache_hits(cache_dirs: Sequence[str], local_dirs: Sequence[str], repo_id, subset=False):
    local_dirs_snapshots = []
    cache_dirs_snapshots = []
    # find all the pre-existing downloads for this repo_id
    try:
        repo_files = set(_hf_fs.ls(repo_id, detail=False))
    except:
        repo_files = []

    if len(repo_files) > 0:
        for local_dir in local_dirs:
            local_path = Path(local_dir) / repo_id
            local_files = frozenset(f"{repo_id}/{f.relative_to(local_path)}" for f in local_path.rglob("*") if f.is_file())
            # fix path representation
            local_files = frozenset(f.replace("\\", "/") for f in local_files)
            # remove .huggingface
            local_files = frozenset(f for f in local_files if not f.startswith(f"{repo_id}/.huggingface") and not f.startswith(f"{repo_id}/.cache"))
            if len(local_files) > 0 and ((subset and local_files.issubset(repo_files)) or (not subset and repo_files.issubset(local_files))):
                local_dirs_snapshots.append(str(local_path))
    else:
        # an empty repository or unknown repository info, trust that if the directory exists, it matches
        for local_dir in local_dirs:
            local_path = Path(local_dir) / repo_id
            if local_path.is_dir():
                local_dirs_snapshots.append(str(local_path))

    for cache_dir in cache_dirs:
        try:
            cache_dirs_snapshots.append(snapshot_download(repo_id, local_files_only=True, cache_dir=cache_dir))
        except FileNotFoundError:
            continue
        except:
            continue
    return cache_dirs_snapshots, local_dirs_snapshots


def _delete_repo_from_huggingface_cache(repo_id: str, cache_dir: Optional[str] = None) -> List[str]:
    results = scan_cache_dir(cache_dir)
    matching = [repo for repo in results.repos if repo.repo_id == repo_id]
    if len(matching) == 0:
        return []
    revisions: List[str] = []
    for repo in matching:
        for revision_info in repo.revisions:
            revisions.append(revision_info.commit_hash)
    results.delete_revisions(*revisions).execute()
    return revisions
