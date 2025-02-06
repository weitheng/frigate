// @ts-nocheck

import { baseUrl } from "@/api/baseUrl";
import ActivityIndicator from "@/components/indicators/activity-indicator";
import LPRDetailDialog from "@/components/overlay/dialog/LPRDetailDialog";
import { Button } from "@/components/ui/button";
import { CamerasFilterButton } from "@/components/filter/CamerasFilterButton";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { Toaster } from "@/components/ui/sonner";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { FrigateConfig } from "@/types/frigateConfig";
import { Event } from "@/types/event";
import { useCallback, useEffect, useMemo, useState } from "react";
import useSWR from "swr";
import { useFormattedTimestamp } from "@/hooks/use-date-utils";
// @ts-ignore
import { LuArrowDownUp, LuTrash2, LuPencil } from "react-icons/lu";
import axios from "axios";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

type SortOption = "score_desc" | "score_asc" | "time_desc" | "time_asc";

// Define an interface for attempt data
interface AttemptData {
  attempt: string;
  plate: string;
  score: string;
  eventId: string | null;
  timestamp: number | null;
}

export default function LPRDebug() {
  const { data: config } = useSWR<FrigateConfig>("config");
  const [sortBy, setSortBy] = useState<SortOption>("time_desc");
  const [selectedCameras, setSelectedCameras] = useState<string[] | undefined>();

  // title
  useEffect(() => {
    document.title = "LPR - Frigate";
  }, []);

  // lpr data
  const { data: lprData, mutate: refreshLPR } = useSWR("lpr/debug");

  const attemptDetails = useMemo(() => {
    if (!lprData) return [];
    return Object.keys(lprData)
      .filter((attempt) => attempt !== 'train')
      .map(a => extractAttemptData(a));
  }, [lprData]);

  const sortedAttempts = useMemo(() => {
    return [...attemptDetails].sort((a, b) => {
      const scoreA = parseFloat(a.score) || 0;
      const scoreB = parseFloat(b.score) || 0;
      const timeA = a.timestamp || 0;
      const timeB = b.timestamp || 0;
      switch(sortBy) {
        case "score_desc": return scoreB - scoreA;
        case "score_asc": return scoreA - scoreB;
        case "time_desc": return timeB - timeA;
        case "time_asc": return timeA - timeB;
        default: return 0;
      }
    });
  }, [attemptDetails, sortBy]);

  const groups = useMemo(() => {
    const grouped: Record<string, typeof sortedAttempts> = {};
    const ungrouped: typeof sortedAttempts = [];
    sortedAttempts.forEach((item) => {
      if (item.plate && !["Recognition Failed", "Raw Capture", "WPOD-NET Detection", "Unknown"].includes(item.plate)) {
        if (!grouped[item.plate]) {
          grouped[item.plate] = [];
        }
        grouped[item.plate].push(item);
      } else {
        ungrouped.push(item);
      }
    });
    Object.keys(grouped).forEach(plate => {
      if (grouped[plate].length < 2) {
        ungrouped.push(...grouped[plate]);
        delete grouped[plate];
      }
    });
    return { grouped, ungrouped };
  }, [sortedAttempts]);

  const tabList = useMemo(() => ["Ungrouped", ...Object.keys(groups.grouped)], [groups]);
  const [currentTab, setCurrentTab] = useState("Ungrouped");
  const [tabNames, setTabNames] = useState<Record<string, string>>({});
  const [renameTab, setRenameTab] = useState<{oldName: string, newName: string} | null>(null);

  useEffect(() => {
    if (renameTab) {
      const newName = window.prompt(`Rename tab '${renameTab.oldName}'`, renameTab.newName);
      if (newName) {
        setTabNames((prev: Record<string, string>) => ({ ...prev, [renameTab.oldName]: newName }));
      }
      setRenameTab(null);
    }
  }, [renameTab]);

  const deleteAllInTab = useCallback(() => {
    let ids: string[] = [];
    if (currentTab === "Ungrouped") {
      ids = groups.ungrouped.map(item => item.attempt);
    } else {
      ids = groups.grouped[currentTab].map(item => item.attempt);
    }
    axios.post(`/lpr/debug/delete`, { ids })
      .then((resp: any) => {
        if (resp.status === 200) {
          toast.success(`Successfully deleted all images in tab.`, { position:"top-center" });
          refreshLPR();
        }
      })
      .catch((error: any) => {
        toast.error(`Failed to delete: ${error.message}`, { position:"top-center" });
      });
  }, [currentTab, groups, refreshLPR]);

  const cameras = useMemo(() => {
    if (!config) return [];
    return Object.keys(config.cameras);
  }, [config]);

  const cameraGroups = useMemo(() => {
    if (!config?.camera_groups) return [];
    return Object.entries(config.camera_groups);
  }, [config]);

  if (!config) {
    return <ActivityIndicator />;
  }

  return (
    <div className="flex size-full flex-col p-2">
      <Toaster />
      <div className="relative mb-2 flex h-11 w-full items-center justify-between">
        <ScrollArea className="w-full whitespace-nowrap">
          <div className="flex flex-row">
            <ScrollBar orientation="horizontal" className="h-0" />
          </div>
        </ScrollArea>
        <div className="flex gap-2">
          <CamerasFilterButton
            allCameras={cameras}
            groups={cameraGroups}
            selectedCameras={selectedCameras}
            updateCameraFilter={setSelectedCameras}
          />
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button className={`flex gap-2 ${sortBy !== "time_desc" ? "select" : "default"}`}>
                <LuArrowDownUp className="size-5" />
                Sort
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              <DropdownMenuLabel>Sort by</DropdownMenuLabel>
              <DropdownMenuItem onClick={() => setSortBy("score_desc")} className={sortBy === "score_desc" ? "bg-accent" : ""}>
                Highest Score First
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortBy("score_asc")} className={sortBy === "score_asc" ? "bg-accent" : ""}>
                Lowest Score First
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortBy("time_desc")} className={sortBy === "time_desc" ? "bg-accent" : ""}>
                Most Recent
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortBy("time_asc")} className={sortBy === "time_asc" ? "bg-accent" : ""}>
                Oldest First
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
      <div className="mb-4">
        <ToggleGroup
          className="*:rounded-md *:px-3 *:py-2"
          type="single"
          value={currentTab}
          onValueChange={(value: string) => setCurrentTab(value)}
        >
          {tabList.map((tab) => (
            <ToggleGroupItem key={tab} value={tab} aria-label={`Select ${tab}`}>
              <span>{tabNames[tab] || tab}</span>
              {tab !== "Ungrouped" && (
                <button onClick={(e) => { e.stopPropagation(); setRenameTab({ oldName: tab, newName: tab }); }}>
                  <LuPencil className="ml-1 size-4" />
                </button>
              )}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
        <div className="mt-2">
          <Button variant="destructive" onClick={deleteAllInTab}>Delete All</Button>
        </div>
      </div>
      <div className="scrollbar-container grid grid-cols-[repeat(auto-fill,minmax(200px,1fr))] gap-2 overflow-y-auto">
        {(currentTab === "Ungrouped" ? groups.ungrouped : groups.grouped[currentTab] || []).map((item: AttemptData) => (
          <LPRAttempt 
            key={item.attempt} 
            attempt={item.attempt} 
            config={config} 
            onRefresh={refreshLPR}
          />
        ))}
      </div>
    </div>
  );
}

type LPRAttemptProps = {
  attempt: string;
  config: FrigateConfig;
  onRefresh: () => void;
};

function LPRAttempt({ attempt, config, onRefresh }: LPRAttemptProps) {
  const [showDialog, setShowDialog] = useState(false);
  const data = useMemo(() => extractAttemptData(attempt), [attempt]);

  const { data: event } = useSWR<Event>(
    data.eventId ? ["event", { id: data.eventId }] : null
  );

  const timestamp = useFormattedTimestamp(
    event?.start_time ?? data.timestamp ?? 0,
    config?.ui.time_format == "24hour" ? "%b %-d %Y, %H:%M" : "%b %-d %Y, %I:%M %p",
    config?.ui.timezone,
  );

  const onDelete = useCallback(() => {
    axios
      .post(`/lpr/debug/delete`, { ids: [attempt] })
      .then((resp) => {
        if (resp.status == 200) {
          toast.success(`Successfully deleted LPR debug image.`, {
            position: "top-center",
          });
          onRefresh();
        }
      })
      .catch((error) => {
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
  }, [attempt, onRefresh]);

  return (
    <>
      <LPRDetailDialog
        open={showDialog}
        setOpen={setShowDialog}
        event={event}
        config={config}
        lprImage={attempt}
        rawImage={`raw_${data.eventId}.jpg`}
      />

      <div className="relative flex flex-col rounded-lg">
        <div className="flex flex-row gap-2 w-full overflow-hidden rounded-t-lg border border-t-0 *:text-card-foreground">
          {/* Main Image (OCR/Raw/WPOD-NET) */}
          <div 
            className="flex-1 cursor-pointer"
            onClick={() => setShowDialog(true)}
          >
            <div className="relative w-full h-40 bg-black flex items-center justify-center">
              <img 
                className="w-full h-full object-cover" 
                src={`${baseUrl}clips/${attempt.startsWith('plate_') ? 'lpd' : 'lpr'}/${attempt}`}
                alt={data.plate}
              />
              {data.timestamp && (
                <div className="absolute bottom-1 left-1 bg-black bg-opacity-50 text-white text-xs p-1 rounded">
                  {(() => {
                    const d = new Date(data.timestamp * 1000);
                    const day = ('0' + d.getDate()).slice(-2);
                    const month = ('0' + (d.getMonth() + 1)).slice(-2);
                    const year = d.getFullYear();
                    return `${day}/${month}/${year}`;
                  })()}
                </div>
              )}
            </div>
          </div>
          {/* Show WPOD-NET detection alongside OCR results */}
          {data.eventId && !attempt.startsWith('plate_') && (
            <div 
              className="flex-1 cursor-pointer"
              onClick={() => setShowDialog(true)}
            >
              <div className="aspect-[2/1] flex items-center justify-center bg-black">
                <img 
                  className="h-40 max-w-none" 
                  src={`${baseUrl}clips/lpd/plate_${data.eventId}.jpg`}
                  alt="WPOD-NET Detection"
                />
              </div>
            </div>
          )}
        </div>
        <div className="flex w-full grow items-center justify-between rounded-b-lg border border-t-0 bg-card p-3 text-card-foreground">
          <div className="flex flex-col items-start text-xs text-primary-variant">
            <div className="capitalize">{data.plate}</div>
            <div className={cn(
              "font-semibold",
              Number(data.score) >= (config?.lpr?.threshold || 0.8) * 100
                ? "text-success"
                : "text-danger"
            )}>
              {data.score === "0" || !data.score ? "No score" : `${data.score}%`}
            </div>
            {event && (
              <div className="text-xs text-muted-foreground">
                {timestamp}
              </div>
            )}
          </div>
          <div className="flex flex-row items-start justify-end gap-5 md:gap-4">
            <Tooltip>
              <TooltipTrigger>
                <LuTrash2
                  className="size-5 cursor-pointer text-primary-variant hover:text-primary"
                  onClick={onDelete}
                />
              </TooltipTrigger>
              <TooltipContent>Delete Image</TooltipContent>
            </Tooltip>
          </div>
        </div>
      </div>
    </>
  );
}

function extractAttemptData(attempt: string): AttemptData {
  if (attempt.startsWith('no_text_')) {
    const timestamp = parseInt(attempt.replace('no_text_', '').replace('.jpg', ''));
    return { attempt, plate: "Recognition Failed", score: "0", eventId: null, timestamp };
  } else if (attempt.startsWith('raw_')) {
    const stripped = attempt.replace('raw_', '').replace('.jpg', '');
    const parts = stripped.split('_');
    if (parts.length === 2) {
      const [eventId, ts] = parts;
      return { attempt, plate: "Raw Capture", score: "0", eventId, timestamp: parseInt(ts) };
    }
    return { attempt, plate: "Raw Capture", score: "0", eventId: stripped, timestamp: null };
  } else if (attempt.startsWith('plate_')) {
    const stripped = attempt.replace('plate_', '').replace('.jpg', '');
    const parts = stripped.split('_');
    if (parts.length === 2) {
      const [eventId, ts] = parts;
      return { attempt, plate: "WPOD-NET Detection", score: "0", eventId, timestamp: parseInt(ts) };
    }
    return { attempt, plate: "WPOD-NET Detection", score: "0", eventId: stripped, timestamp: null };
  } else {
    // Remove the .jpg suffix and split by underscore
    const parts = attempt.replace('.jpg', '').split('_');
    if (parts.length === 3) {
      // Format: plate_score_timestamp.jpg
      const [plate, score, ts] = parts;
      return { attempt, plate, score, eventId: null, timestamp: parseInt(ts) };
    } else if (parts.length === 4) {
      // Format: plate_score_eventId_timestamp.jpg
      const [plate, score, eventId, ts] = parts;
      return { attempt, plate, score, eventId, timestamp: parseInt(ts) };
    } else {
      return { attempt, plate: "Unknown", score: "0", eventId: null, timestamp: null };
    }
  }
} 