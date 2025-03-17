import { baseUrl } from "@/api/baseUrl";
import TimeAgo from "@/components/dynamic/TimeAgo";
import AddFaceIcon from "@/components/icons/AddFaceIcon";
import ActivityIndicator from "@/components/indicators/activity-indicator";
import CreateFaceWizardDialog from "@/components/overlay/detail/FaceCreateWizardDialog";
import UploadImageDialog from "@/components/overlay/dialog/UploadImageDialog";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { Toaster } from "@/components/ui/sonner";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import useKeyboardListener from "@/hooks/use-keyboard-listener";
import useOptimisticState from "@/hooks/use-optimistic-state";
import { cn } from "@/lib/utils";
import { FrigateConfig } from "@/types/frigateConfig";
import axios, { AxiosResponse, AxiosError } from "axios";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { isDesktop } from "react-device-detect";
import { useTranslation } from "react-i18next";
import { LuImagePlus, LuRefreshCw, LuScanFace, LuTrash2, LuArrowLeft, LuArrowRight } from "react-icons/lu";
import { toast } from "sonner";
import useSWR from "swr";

export default function FaceLibrary() {
  const { t } = useTranslation(["views/faceLibrary"]);

  const { data: config } = useSWR<FrigateConfig>("config");

  // title

  useEffect(() => {
    document.title = t("documentTitle");
  }, [t]);

  const [page, setPage] = useState<string>();
  const [pageToggle, setPageToggle] = useOptimisticState(page, setPage, 100);
  const tabsRef = useRef<HTMLDivElement | null>(null);

  // face data

  const { data: faceData, mutate: refreshFaces } = useSWR("faces");

  const faces = useMemo<string[]>(
    () =>
      faceData ? Object.keys(faceData).filter((face) => face != "train") : [],
    [faceData],
  );
  const faceImages = useMemo<string[]>(
    () => (pageToggle && faceData ? faceData[pageToggle] : []),
    [pageToggle, faceData],
  );

  const trainImages = useMemo<string[]>(
    () => faceData?.["train"] || [],
    [faceData],
  );

  // Pagination state
  const [currentPage, setCurrentPage] = useState<number>(1);
  const itemsPerPage = 100; // Define items per page

  // Calculate total pages based on active content
  const totalPages = useMemo(() => {
    const totalItems = pageToggle === "train" ? trainImages.length : faceImages.length;
    return Math.ceil(totalItems / itemsPerPage);
  }, [pageToggle, trainImages, faceImages, itemsPerPage]);

  // Get current items for both grids
  const currentFaceImages = useMemo<string[]>(() => {
    const indexOfLastItem = currentPage * itemsPerPage;
    const indexOfFirstItem = indexOfLastItem - itemsPerPage;
    return faceImages.slice(indexOfFirstItem, indexOfLastItem);
  }, [currentPage, faceImages, itemsPerPage]);

  const currentTrainImages = useMemo<string[]>(() => {
    const indexOfLastItem = currentPage * itemsPerPage;
    const indexOfFirstItem = indexOfLastItem - itemsPerPage;
    return trainImages.slice(indexOfFirstItem, indexOfLastItem);
  }, [currentPage, trainImages, itemsPerPage]);

  useEffect(() => {
    if (!pageToggle) {
      if (trainImages.length > 0) {
        setPageToggle("train");
      } else if (faces) {
        setPageToggle(faces[0]);
      }
    } else if (pageToggle == "train" && trainImages.length == 0) {
      setPageToggle(faces[0]);
    }
    // Reset current page when switching tabs
    setCurrentPage(1); // Reset to page 1 when changing tabs
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainImages, faces, pageToggle]);

  // upload

  const [upload, setUpload] = useState(false);
  const [addFace, setAddFace] = useState(false);

  const onUploadImage = useCallback(
    (file: File) => {
      const formData = new FormData();
      formData.append("file", file);
      axios
        .post(`faces/${pageToggle}/register`, formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        })
        .then((resp) => {
          if (resp.status == 200) {
            setUpload(false);
            refreshFaces();
            toast.success(t("toast.success.uploadedImage"), {
              position: "top-center",
            });
          }
        })
        .catch((error) => {
          const errorMessage =
            error.response?.data?.message ||
            error.response?.data?.detail ||
            "Unknown error";
          toast.error(t("toast.error.uploadingImageFailed", { errorMessage }), {
            position: "top-center",
          });
        });
    },
    [pageToggle, refreshFaces, t],
  );

  // face multiselect

  const [selectedFaces, setSelectedFaces] = useState<string[]>([]);

  const onClickFace = useCallback(
    (imageId: string, ctrl: boolean) => {
      if (selectedFaces.length == 0 && !ctrl) {
        return;
      }

      const index = selectedFaces.indexOf(imageId);

      if (index != -1) {
        if (selectedFaces.length == 1) {
          setSelectedFaces([]);
        } else {
          const copy = [
            ...selectedFaces.slice(0, index),
            ...selectedFaces.slice(index + 1),
          ];
          setSelectedFaces(copy);
        }
      } else {
        const copy = [...selectedFaces];
        copy.push(imageId);
        setSelectedFaces(copy);
      }
    },
    [selectedFaces, setSelectedFaces],
  );

  const onDelete = useCallback(
    (name: string, ids: string[]) => {
      axios
        .post(`/faces/${name}/delete`, { ids })
        .then((resp) => {
          setSelectedFaces([]);

          if (resp.status == 200) {
            toast.success(t("toast.success.deletedFace"), {
              position: "top-center",
            });

            if (faceImages.length == 1) {
              // face has been deleted
              setPageToggle("");
            }

            refreshFaces();
          }
        })
        .catch((error) => {
          const errorMessage =
            error.response?.data?.message ||
            error.response?.data?.detail ||
            "Unknown error";
          toast.error(t("toast.error.deleteFaceFailed", { errorMessage }), {
            position: "top-center",
          });
        });
    },
    [faceImages, refreshFaces, setPageToggle, t],
  );

  // keyboard

  useKeyboardListener(["a", "Escape"], (key, modifiers) => {
    if (modifiers.repeat || !modifiers.down) {
      return;
    }

    switch (key) {
      case "a":
        if (modifiers.ctrl) {
          setSelectedFaces([...trainImages]);
        }
        break;
      case "Escape":
        setSelectedFaces([]);
        break;
    }
  });

  // Add handleDeleteAll callback
  const handleDeleteAll = useCallback(() => {
    const imagesToDelete = pageToggle === "train" ? currentTrainImages : currentFaceImages;

    axios
      .post(`/faces/${pageToggle}/delete`, { ids: imagesToDelete })
      .then((resp: AxiosResponse) => {
        if (resp.status === 200) {
          toast.success(`Successfully deleted all images on this page.`, {
            position: "top-center",
          });
          refreshFaces();
          if (faceImages.length === 0) {
            setPageToggle("train");
          }
        }
      })
      .catch((error: AxiosError<{ message: string }>) => {
        if (error.response?.data?.message) {
          toast.error(`Failed to delete: ${error.response.data.message}`, {
            position: "top-center",
          });
        } else {
          toast.error(`Failed to delete: ${error.message}`, {
            position: "top-center",
          });
        }
      });
  }, [pageToggle, currentTrainImages, currentFaceImages, refreshFaces, faceImages]);

  if (!config) {
    return <ActivityIndicator />;
  }

  return (
    <div className="flex size-full flex-col p-2">
      <Toaster />

      <UploadImageDialog
        open={upload}
        title={t("uploadFaceImage.title")}
        description={t("uploadFaceImage.desc", { pageToggle })}
        setOpen={setUpload}
        onSave={onUploadImage}
      />

      <CreateFaceWizardDialog
        open={addFace}
        setOpen={setAddFace}
        onFinish={refreshFaces}
      />

      <div className="relative mb-2 flex h-11 w-full items-center justify-between">
        <ScrollArea className="w-full whitespace-nowrap">
          <div ref={tabsRef} className="flex flex-row">
            <ToggleGroup
              className="*:rounded-md *:px-3 *:py-4"
              type="single"
              size="sm"
              value={pageToggle}
              onValueChange={(value: string) => {
                if (value) {
                  setPageToggle(value);
                }
              }}
            >
              {trainImages.length > 0 && (
                <>
                  <ToggleGroupItem
                    value="train"
                    className={`flex scroll-mx-10 items-center justify-between gap-2 ${pageToggle == "train" ? "" : "*:text-muted-foreground"}`}
                    data-nav-item="train"
                    aria-label={t("train.aria")}
                  >
                    <div>{t("train.title")}</div>
                  </ToggleGroupItem>
                  <div>|</div>
                </>
              )}

              {Object.values(faces).map((item) => (
                <ToggleGroupItem
                  key={item}
                  className={`flex scroll-mx-10 items-center justify-between gap-2 ${pageToggle == item ? "" : "*:text-muted-foreground"}`}
                  value={item}
                  data-nav-item={item}
                  aria-label={t("selectItem", { item })}
                >
                  <div className="capitalize">
                    {item} ({faceData[item].length})
                  </div>
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
            <ScrollBar orientation="horizontal" className="h-0" />
          </div>
        </ScrollArea>
        {selectedFaces?.length > 0 ? (
          <div className="flex items-center justify-center gap-2">
            <Button
              className="flex gap-2"
              onClick={() => onDelete("train", selectedFaces)}
            >
              <LuTrash2 className="size-7 rounded-md p-1 text-secondary-foreground" />
              {isDesktop && t("button.deleteFaceAttempts")}
            </Button>
          </div>
        ) : (
          <div className="flex items-center justify-center gap-2">
            <Button className="flex gap-2" onClick={() => setAddFace(true)}>
              <LuScanFace className="size-7 rounded-md p-1 text-secondary-foreground" />
              {isDesktop && t("button.addFace")}
            </Button>
            {pageToggle != "train" && (
              <Button className="flex gap-2" onClick={() => setUpload(true)}>
                <LuImagePlus className="size-7 rounded-md p-1 text-secondary-foreground" />
                {isDesktop && t("button.uploadImage")}
              </Button>
            )}
          </div>
        )}
      </div>

      {/* Move Delete All Button below the tabs */}
      <div className="mb-2">
        <Button onClick={handleDeleteAll} variant="destructive">
          Delete All
        </Button>
      </div>

      {pageToggle &&
        (pageToggle == "train" ? (
          <TrainingGrid
            config={config}
            attemptImages={currentTrainImages}
            faceNames={faces}
            selectedFaces={selectedFaces}
            onClickFace={onClickFace}
            onRefresh={refreshFaces}
          />
        ) : (
          <FaceGrid
            faceImages={currentFaceImages}
            pageToggle={pageToggle}
            onDelete={onDelete}
          />
        ))}

      {/* Pagination Controls */}
      <div className="flex items-center justify-center mt-4">
        <button
          onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
          disabled={currentPage === 1}
          className={`p-2 ${currentPage === 1 ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <LuArrowLeft className="text-gray-600" />
        </button>
        <span className="text-sm text-gray-700 mx-2">
          Page {currentPage} of {totalPages}
        </span>
        <button
          onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
          disabled={currentPage === totalPages}
          className={`p-2 ${currentPage === totalPages ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <LuArrowRight className="text-gray-600" />
        </button>
      </div>
    </div>
  );
}

type TrainingGridProps = {
  config: FrigateConfig;
  attemptImages: string[];
  faceNames: string[];
  selectedFaces: string[];
  onClickFace: (image: string, ctrl: boolean) => void;
  onRefresh: () => void;
};
function TrainingGrid({
  config,
  attemptImages,
  faceNames,
  selectedFaces,
  onClickFace,
  onRefresh,
}: TrainingGridProps) {
  return (
    <div className="scrollbar-container flex flex-wrap gap-2 overflow-y-scroll p-1">
      {attemptImages.map((image: string) => (
        <FaceAttempt
          key={image}
          image={image}
          faceNames={faceNames}
          threshold={config.face_recognition.recognition_threshold}
          selected={selectedFaces.includes(image)}
          onClick={(meta) => onClickFace(image, meta)}
          onRefresh={onRefresh}
        />
      ))}
    </div>
  );
}

type FaceAttemptProps = {
  image: string;
  faceNames: string[];
  threshold: number;
  selected: boolean;
  onClick: (meta: boolean) => void;
  onRefresh: () => void;
};
function FaceAttempt({
  image,
  faceNames,
  threshold,
  selected,
  onClick,
  onRefresh,
}: FaceAttemptProps) {
  const { t } = useTranslation(["views/faceLibrary"]);
  const data = useMemo(() => {
    const parts = image.split("-");

    return {
      timestamp: Number.parseFloat(parts[0]),
      eventId: `${parts[0]}-${parts[1]}`,
      name: parts[2],
      score: parts[3],
    };
  }, [image]);

  const onTrainAttempt = useCallback(
    (trainName: string) => {
      axios
        .post(`/faces/train/${trainName}/classify`, { training_file: image })
        .then((resp) => {
          if (resp.status == 200) {
            toast.success(t("toast.success.trainedFace"), {
              position: "top-center",
            });
            onRefresh();
          }
        })
        .catch((error) => {
          const errorMessage =
            error.response?.data?.message ||
            error.response?.data?.detail ||
            "Unknown error";
          toast.error(t("toast.error.trainFailed", { errorMessage }), {
            position: "top-center",
          });
        });
    },
    [image, onRefresh, t],
  );

  const onReprocess = useCallback(() => {
    axios
      .post(`/faces/reprocess`, { training_file: image })
      .then((resp) => {
        if (resp.status == 200) {
          toast.success(t("toast.success.updatedFaceScore"), {
            position: "top-center",
          });
          onRefresh();
        }
      })
      .catch((error) => {
        const errorMessage =
          error.response?.data?.message ||
          error.response?.data?.detail ||
          "Unknown error";
        toast.error(t("toast.error.updateFaceScoreFailed", { errorMessage }), {
          position: "top-center",
        });
      });
  }, [image, onRefresh, t]);

  return (
    <div
      className={cn(
        "relative flex cursor-pointer flex-col rounded-lg outline outline-[3px]",
        selected
          ? "shadow-selected outline-selected"
          : "outline-transparent duration-500",
      )}
      onClick={(e) => onClick(e.metaKey || e.ctrlKey)}
    >
      <div className="relative w-full overflow-hidden rounded-t-lg border border-t-0 *:text-card-foreground">
        <img className="size-44" src={`${baseUrl}clips/faces/train/${image}`} />
        <div className="absolute bottom-1 right-1 z-10 rounded-lg bg-black/50 px-2 py-1 text-xs text-white">
          <TimeAgo time={data.timestamp * 1000} dense />
        </div>
      </div>
      <div className="rounded-b-lg bg-card p-2">
        <div className="flex w-full flex-row items-center justify-between gap-2">
          <div className="flex flex-col items-start text-xs text-primary-variant">
            <div className="capitalize">{data.name}</div>
            <div
              className={cn(
                Number.parseFloat(data.score) >= threshold
                  ? "text-success"
                  : "text-danger",
              )}
            >
              {Number.parseFloat(data.score) * 100}%
            </div>
          </div>
          <div className="flex flex-row items-start justify-end gap-5 md:gap-4">
            <Tooltip>
              <DropdownMenu>
                <DropdownMenuTrigger>
                  <TooltipTrigger>
                    <AddFaceIcon className="size-5 cursor-pointer text-primary-variant hover:text-primary" />
                  </TooltipTrigger>
                </DropdownMenuTrigger>
                <DropdownMenuContent>
                  <DropdownMenuLabel>{t("trainFaceAs")}</DropdownMenuLabel>
                  {faceNames.map((faceName) => (
                    <DropdownMenuItem
                      key={faceName}
                      className="cursor-pointer capitalize"
                      onClick={() => onTrainAttempt(faceName)}
                    >
                      {faceName}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
              <TooltipContent>{t("trainFaceAsPerson")}</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger>
                <LuRefreshCw
                  className="size-5 cursor-pointer text-primary-variant hover:text-primary"
                  onClick={() => onReprocess()}
                />
              </TooltipTrigger>
              <TooltipContent>{t("button.reprocessFace")}</TooltipContent>
            </Tooltip>
          </div>
        </div>
      </div>
    </div>
  );
}

type FaceGridProps = {
  faceImages: string[];
  pageToggle: string;
  onDelete: (name: string, ids: string[]) => void;
};
function FaceGrid({ faceImages, pageToggle, onDelete }: FaceGridProps) {
  return (
    <div className="scrollbar-container flex flex-wrap gap-2 overflow-y-scroll">
      {faceImages.map((image: string) => (
        <FaceImage
          key={image}
          name={pageToggle}
          image={image}
          onDelete={onDelete}
        />
      ))}
    </div>
  );
}

type FaceImageProps = {
  name: string;
  image: string;
  onDelete: (name: string, ids: string[]) => void;
};
function FaceImage({ name, image, onDelete }: FaceImageProps) {
  const { t } = useTranslation(["views/faceLibrary"]);

  return (
    <div className="relative flex flex-col rounded-lg">
      <div className="w-full overflow-hidden rounded-t-lg border border-t-0 *:text-card-foreground">
        <img className="h-40" src={`${baseUrl}clips/faces/${name}/${image}`} />
      </div>
      <div className="rounded-b-lg bg-card p-2">
        <div className="flex w-full flex-row items-center justify-between gap-2">
          <div className="flex flex-col items-start text-xs text-primary-variant">
            <div className="capitalize">{name}</div>
          </div>
          <div className="flex flex-row items-start justify-end gap-5 md:gap-4">
            <Tooltip>
              <TooltipTrigger>
                <LuTrash2
                  className="size-5 cursor-pointer text-primary-variant hover:text-primary"
                  onClick={() => onDelete(name, [image])}
                />
              </TooltipTrigger>
              <TooltipContent>{t("button.deleteFaceAttempts")}</TooltipContent>
            </Tooltip>
          </div>
        </div>
      </div>
    </div>
  );
}
